// resa_3plus1.cpp
// Replica-Exchange (Parallel Tempering) for 3+1 neutrino oscillation fit
// g++ -O2 -std=c++17 resa_3plus1.cpp -o resa

#include <bits/stdc++.h>
using namespace std;
using cd = complex<double>;
const double PI = acos(-1.0);
const cd I(0.0,1.0);

// ---------- Utilities ----------
double sq(double x){ return x*x; }
double deg2rad(double d){ return d * PI/180.0; }

// RNG
std::mt19937_64 rng((uint64_t)chrono::high_resolution_clock::now().time_since_epoch().count());
double uniform01(){ return std::uniform_real_distribution<double>(0.0,1.0)(rng); }
double gaussian(double sigma){ return std::normal_distribution<double>(0.0,sigma)(rng); }

// ---------- Build 4x4 mixing matrix U(angles, delta24) ----------
// We apply sequential rotations R12, R13, R14, R23, R24(with phase), R34
using Mat = vector<vector<cd>>;
Mat eye4(){
    Mat A(4, vector<cd>(4, cd(0.0,0.0)));
    for(int i=0;i<4;i++) A[i][i]=1.0;
    return A;
}
Mat matMul(const Mat &A, const Mat &B){
    Mat C(4, vector<cd>(4, cd(0.0,0.0)));
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        cd s=0;
        for(int k=0;k<4;k++) s += A[i][k]*B[k][j];
        C[i][j]=s;
    }
    return C;
}
Mat rotation_ij(int i, int j, double theta, double phase=0.0){
    // rotation in (i,j) plane (0-based indices). If phase != 0, multiply
    // the sin term by exp(-i phase) for one side and exp(i phase) for the other
    Mat R = eye4();
    double c = cos(theta);
    double s = sin(theta);
    cd e_minus = std::exp(-I * phase);
    cd e_plus  = std::exp( I * phase);
    // We'll implement complex rotation such that:
    // R_ii = c, R_jj = c, R_ij = s * e_minus, R_ji = -s * e_plus
    R[i][i] = c;
    R[j][j] = c;
    R[i][j] = cd(s,0.0) * e_minus;
    R[j][i] = cd(-s,0.0) * e_plus;
    return R;
}

// angles (radians) and delta24 (radians). angles order:
// theta12, theta13, theta14, theta23, theta24, theta34
Mat buildU(const array<double,6> &angles, double delta24){
    // sequence: R12 * R13 * R14 * R23 * R24(with phase) * R34
    Mat U = eye4();
    vector<pair<pair<int,int>, pair<double,double>>> seq = {
        {{0,1}, {angles[0], 0.0}},
        {{0,2}, {angles[1], 0.0}},
        {{0,3}, {angles[2], 0.0}},
        {{1,2}, {angles[3], 0.0}},
        {{1,3}, {angles[4], delta24}},
        {{2,3}, {angles[5], 0.0}}
    };
    for(auto &step : seq){
        int i = step.first.first;
        int j = step.first.second;
        double th = step.second.first;
        double ph = step.second.second;
        Mat R = rotation_ij(i, j, th, ph);
        U = matMul(U, R);
    }
    return U;
}

// ---------- Oscillation probability P_{alpha->beta}(E,L) in vacuum ----------
// alpha, beta in {0=e,1=mu,2=tau,3=s}
// mass splittings: dm21, dm31, dm41 (all in eV^2). m1^2=0, m2^2=dm21, m3^2=dm31, m4^2=dm41
double oscProb(const Mat &U, int alpha, int beta, double E_GeV, double L_km,
               double dm21, double dm31, double dm41)
{
    // Convert units: common constant for phase: Delta = 1.267 * dm2[eV^2] * L[km] / E[GeV]
    array<double,4> m2 = {0.0, dm21, dm31, dm41};
    cd amp = 0.0;
    // amplitude sum_i U_{alpha i} * conj(U_{beta i}) * exp(-i*phi_i)
    for(int i=0;i<4;i++){
        double phi = 1.267 * m2[i] * L_km / E_GeV; // radians
        cd phase = std::exp(-I * phi);
        amp += U[alpha][i] * std::conj(U[beta][i]) * phase;
    }
    double prob = norm(amp); // |amp|^2
    // Numerical safety
    if(prob < 0) prob = 0;
    if(prob > 1) prob = 1;
    return prob;
}

// ---------- Chi-square / likelihood ----------
// We'll assume user has arrays energies, baselines, observed probabilities (or counts) and errors.
// The model predicts a probability; if data are counts you should convert appropriately (or use Poisson likelihood)
struct DataPoint {
    double E;      // GeV
    double L;      // km
    int alpha;     // initial flavor index (0=e,1=mu,...)
    int beta;      // final flavor index
    double obs;    // observed probability (or event rate normalized)
    double err;    // 1-sigma uncertainty
};

double chi2_for_params(const vector<DataPoint> &data,
                       const array<double,6> &angles, double delta24,
                       double dm21, double dm31, double dm41)
{
    Mat U = buildU(angles, delta24);
    double chi2 = 0.0;
    for(const auto &d : data){
        double pred = oscProb(U, d.alpha, d.beta, d.E, d.L, dm21, dm31, dm41);
        double res = (d.obs - pred);
        double sig = max(d.err, 1e-6);
        chi2 += (res*res) / (sig*sig);
    }
    return chi2;
}

// ---------- Parameter container & proposal ----------
struct Params {
    // ordering for algorithmic convenience:
    // [dm21, dm31, dm41, th12, th13, th14, th23, th24, th34, delta24]
    double dm21, dm31, dm41;
    array<double,6> angles;
    double delta24;
};

Params random_init(){
    Params p;
    // default plausible ranges (set reasonably; user should tune)
    p.dm21 = 7.5e-5; // typical solar (placeholder)
    p.dm31 = 2.5e-3;  // placeholder
    p.dm41 = 1.0;    // try 1 eV^2 as typical sterile region guess
    // angles in radians: typical ranges 0..pi/2
    p.angles = {deg2rad(33.44), deg2rad(8.57), deg2rad(2.0), deg2rad(49.2), deg2rad(5.0), deg2rad(10.0)};
    p.delta24 = 0.0;
    return p;
}

Params perturb(const Params &p, const vector<double> &step_scales){
    Params q = p;
    // step_scales vector should map to each parameter scale (same ordering)
    // add gaussian perturbation scaled by step_scales
    q.dm21 += gaussian(step_scales[0]);
    q.dm31 += gaussian(step_scales[1]);
    q.dm41 += gaussian(step_scales[2]);
    for(int i=0;i<6;i++) q.angles[i] += gaussian(step_scales[3+i]);
    q.delta24 += gaussian(step_scales[9]);
    // enforce sensible bounds:
    q.dm21 = max(1e-6, q.dm21);
    q.dm31 = max(1e-6, q.dm31);
    q.dm41 = max(1e-6, q.dm41);
    for(int i=0;i<6;i++){
        // keep angles in [0, pi/2] for mixing angles
        if(q.angles[i] < 0) q.angles[i] = -q.angles[i];
        if(q.angles[i] > PI/2) q.angles[i] = fmod(q.angles[i], PI/2);
    }
    // phase in [-pi, pi]
    if(q.delta24 > PI) q.delta24 = fmod(q.delta24 + PI, 2*PI) - PI;
    if(q.delta24 < -PI) q.delta24 = fmod(q.delta24 - PI, 2*PI) + PI;
    return q;
}

// ---------- Replica structure ----------
struct Replica {
    Params params;
    double T;      // temperature
    double chi2;   // current chi2
};

// ---------- Parallel tempering routine (serial implementation) ----------
void run_replica_exchange(
    vector<Replica> &reps,
    const vector<DataPoint> &data,
    const vector<double> &step_scales,
    int nsteps_each_replica,
    int swap_interval,
    int rng_seed = 0
){
    if(rng_seed) rng.seed((uint64_t)rng_seed);
    int N = reps.size();
    // initial chi2 compute
    for(int i=0;i<N;i++){
        const auto &p = reps[i].params;
        reps[i].chi2 = chi2_for_params(data, p.angles, p.delta24, p.dm21, p.dm31, p.dm41);
    }

    // main loop: perform many local Metropolis updates per replica and occasional swaps
    for(int step=0; step < nsteps_each_replica; ++step){
        // Update each replica with local Metropolis moves
        for(int r=0;r<N;r++){
            // propose
            Params prop = perturb(reps[r].params, step_scales);
            double chi2_prop = chi2_for_params(data, prop.angles, prop.delta24, prop.dm21, prop.dm31, prop.dm41);
            double dE = chi2_prop - reps[r].chi2;
            double beta = 1.0 / reps[r].T;
            double accept_prob = exp(-beta * dE);
            if(dE <= 0 || uniform01() < accept_prob){
                reps[r].params = prop;
                reps[r].chi2 = chi2_prop;
            }
        }

        // attempt swaps between adjacent replicas every swap_interval steps
        if( (step % swap_interval) == 0 ){
            for(int r=0; r < N-1; ++r){
                // swap r <-> r+1
                double beta1 = 1.0 / reps[r].T;
                double beta2 = 1.0 / reps[r+1].T;
                double chi1 = reps[r].chi2;
                double chi2 = reps[r+1].chi2;
                // acceptance prob: min(1, exp( (beta1 - beta2)*(chi2 - chi1) ) )
                double ex = (beta1 - beta2) * (chi2 - chi1);
                double acc = (ex >= 0) ? 1.0 : exp(ex); // careful overflow
                if(uniform01() < acc){
                    swap(reps[r].params, reps[r+1].params);
                    swap(reps[r].chi2, reps[r+1].chi2);
                }
            }
        }
    }
}

// ---------- Example usage with placeholder data ----------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ---------- Example placeholder dataset ----------
    // Replace these with your experimental energies, baselines, observed probabilities and errors.
    vector<DataPoint> data;
    // Suppose we probe mu->e appearance at several energies for a fixed baseline L:
    double L = 1.0; // km (replace with your baseline)
    for(double E=0.5; E<=10.0; E += 0.5){
        DataPoint d;
        d.E = E;
        d.L = L;
        d.alpha = 1; // muon neutrino initial
        d.beta  = 0; // electron neutrino final
        // For demonstration: assume zero observed (no appearance), error 0.01
        d.obs = 0.01 * uniform01(); // small random noise placeholder
        d.err = 0.02;
        data.push_back(d);
    }

    // ---------- Replica setup ----------
    int Nrep = 8;
    vector<double> temps(Nrep);
    // geometric ladder: T0 = 1.0 (target), Tmax = 50.0
    double Tmin = 1.0;
    double Tmax = 50.0;
    for(int i=0;i<Nrep;i++){
        double frac = (double)i / (double)(Nrep-1);
        temps[i] = Tmin * pow(Tmax/Tmin, frac);
    }

    // Initialize replicas
    vector<Replica> reps(Nrep);
    for(int i=0;i<Nrep;i++){
        reps[i].params = random_init();
        reps[i].T = temps[i];
    }

    // Step scales for perturbation (order: dm21, dm31, dm41, th12..th34, delta24)
    vector<double> step_scales = {
        1e-6, 5e-4, 0.1,   // dm21, dm31, dm41
        deg2rad(0.5), deg2rad(0.2), deg2rad(0.2), // th12, th13, th14
        deg2rad(0.5), deg2rad(0.5), deg2rad(0.5), // th23, th24, th34
        deg2rad(10.0) // delta24
    };

    // Run parallel tempering
    int ncycles = 2000;       // number of "local step" cycles
    int swap_interval = 10;   // try swaps every 10 cycles
    cout << "Running " << ncycles << " cycles, " << Nrep << " replicas...\n";
    run_replica_exchange(reps, data, step_scales, ncycles, swap_interval);

    // find best replica (lowest chi2)
    double best_chi2 = 1e300;
    Params best;
    for(auto &r : reps){
        if(r.chi2 < best_chi2){
            best_chi2 = r.chi2;
            best = r.params;
        }
    }

    cout << "Best chi2 = " << best_chi2 << "\n";
    cout << "Best parameters:\n";
    cout << "dm21 = " << best.dm21 << " eV^2\n";
    cout << "dm31 = " << best.dm31 << " eV^2\n";
    cout << "dm41 = " << best.dm41 << " eV^2\n";
    cout << "angles (deg): ";
    for(int i=0;i<6;i++) cout << (best.angles[i]*180.0/PI) << (i==5? "\n" : ", ");
    cout << "delta24 (deg) = " << (best.delta24*180.0/PI) << "\n";

    // You should save chains / best-fit and run local optimizer to refine
    return 0;
}

