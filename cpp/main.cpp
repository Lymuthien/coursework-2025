#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Args {
    int n_samples = 20000;
    int n_features = 128;
    int iters = 50;
    double lr = 1e-3;
    int threads = 0;
    int seed = 42;
    std::string csv_path = "results.csv";
    std::string label = "CPU";
    std::string per_iter_log = "";
    double target_sec = 0.0;
    int pilot_iters = 5;
    int runs = 1;
    bool log_header_force = false;
};

struct Data {
    int N=0, D=0;
    std::vector<double> X, y, w, w_true, r, grad;
};

static inline std::string now_str() {
    std::time_t now = std::time(nullptr); std::tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    return std::string(buf);
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int i, const std::string& k, int argc) {
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for %s\n", k.c_str());
            exit(1);
        }
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--n-samples") {
            need(i, k, argc);
            a.n_samples = std::stoi(argv[++i]);
        }
        else if (k == "--n-features") {
            need(i, k, argc);
            a.n_features = std::stoi(argv[++i]);
        }
        else if (k == "--iters") {
            need(i, k, argc);
            a.iters = std::stoi(argv[++i]);
        }
        else if (k == "--lr") {
            need(i, k, argc);
            a.lr = std::stod(argv[++i]);
        }
        else if (k == "--threads") { 
            need(i, k, argc); 
            a.threads = std::stoi(argv[++i]);
        }
        else if (k == "--seed") { 
            need(i, k, argc); 
            a.seed = std::stoi(argv[++i]); 
        }
        else if (k == "--csv") { 
            need(i, k, argc); 
            a.csv_path = argv[++i];
        }
        else if (k == "--label") { 
            need(i, k, argc);
            a.label = argv[++i]; 
        }
        else if (k == "--per-iter-log") { 
            need(i, k, argc); 
            a.per_iter_log = argv[++i]; 
        }
        else if (k == "--target-sec") {
            need(i, k, argc);
            a.target_sec = std::stod(argv[++i]);
        }
        else if (k == "--pilot-iters") {
            need(i, k, argc);
            a.pilot_iters = std::stoi(argv[++i]);
        }
        else if (k == "--runs") { 
            need(i, k, argc);
            a.runs = std::stoi(argv[++i]); 
        }
        else if (k == "--log-header-force") {
            a.log_header_force = true;
        }
        else { 
            fprintf(stderr, "Unknown arg: %s\n", k.c_str()); exit(1);
        }
    }
    return a;
}

#ifdef _OPENMP
static inline int eff_threads(int req) {
    return (req > 0) ? req : omp_get_max_threads();
}
#else
static inline int eff_threads(int req) { 
    (void)req; 
    return 1; 
}
#endif

void generate_data(Data& d, int N, int D, int seed) {
    d.N = N;
    d.D = D;
    d.X.assign((size_t)N * D, 0.0);
    d.y.assign(N, 0.0);
    d.w.assign(D, 0.0);
    d.w_true.assign(D, 0.0);
    d.r.assign(N, 0.0);
    d.grad.assign(D, 0.0);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int j = 0; j < D; ++j)
        d.w_true[j] = nd(rng);
    for (int i = 0; i < N; ++i) {
        double yi = 0.0;
        for (int j = 0; j < D; ++j) {
            double x = nd(rng);
            d.X[(size_t)i * D + j] = x;
            yi += x * d.w_true[j];
        }
        d.y[i] = yi + 0.01 * nd(rng);
    }
}

double compute_loss(const Data& d, const std::vector<double>& w) {
    int N = d.N, D = d.D;
    double sse = 0.0;
#pragma omp parallel for reduction(+:sse) if (N > 1000)
    for (int i = 0; i < N; ++i) {
        double pred = 0.0;
        const double* Xi = &d.X[(size_t)i * D];
        for (int j = 0; j < D; ++j)
            pred += Xi[j] * w[j];
        double diff = pred - d.y[i];
        sse += diff * diff;
    }
    return sse / N;
}

double one_iter(Data& d, std::vector<double>& w, double lr, bool need_loss, double* out_loss) {
    auto t0 = std::chrono::high_resolution_clock::now();
    double sse = compute_loss(d, w);
    int N = d.N;  
    int D = d.D;

    std::fill(d.grad.begin(), d.grad.end(), 0.0);
#pragma omp parallel
    {
        std::vector<double> gl(D, 0.0);
#pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            const double* Xi = &d.X[(size_t)i * D];
            double ri = d.r[i];
            for (int j = 0; j < D; ++j)
                gl[j] += Xi[j] * ri;
        }
#pragma omp critical
        {
            for (int j = 0; j < D; ++j)
                d.grad[j] += gl[j];
        }
    }
    double scale = 2.0 / N;
    for (int j = 0; j < D; ++j)
        d.grad[j] *= scale;
    for (int j = 0; j < D; ++j)
        w[j] -= lr * d.grad[j];
    auto t1 = std::chrono::high_resolution_clock::now();
    if (out_loss)
        //*out_loss = need_loss ? (sse / N) : std::numeric_limits<double>::quiet_NaN();
        *out_loss = need_loss ? sse : std::numeric_limits<double>::quiet_NaN();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int calibrate_iters(Data& d, std::vector<double>& w, double lr, int pilot, double target) {
    if (target <= 0)
        return -1;
    int K = std::max(1, pilot);
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        double L;
        sum += one_iter(d, w, lr, false, &L);
    }
    double per = sum / K;
    int need = std::max(1, (int)std::ceil((target * 1000.0) / per));
    return need;
}

void ensure_header(const std::string& path, const std::string& hdr) {
    std::ifstream in(path);
    if (!in.good()) {
        std::ofstream out(path);
        out << hdr << "\n";
    }
}

struct RunResult {
    double total_ms = 0.0, per_iter_ms = 0.0, final_loss = 0.0;
};

RunResult run_once(
    Data& d, std::vector<double>& w,
    double lr,
    int iters,
    const std::string& label,
    int threads,
    const std::string& per_path,
    bool force_header,
    int run_id
) {
    (void)compute_loss(d, w);
    std::ofstream perofs;

    if (!per_path.empty()) {
        std::remove(per_path.c_str());

        bool header = force_header || !std::ifstream(per_path).good();
        perofs.open(per_path, std::ios::app);
        if (header)
            perofs << "date,cpu_label,threads,n_samples,n_features,run_id,iter_idx,time_ms,loss\n";
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string date = now_str();
    double last_loss = 0.0;
    for (int t = 0; t < iters; ++t) {
        double L;
        double ms = one_iter(d, w, lr, !per_path.empty(), &L);
        last_loss = L;
        if (perofs.good()) {
            perofs << date << "," << label << "," << threads << "," << d.N << "," << d.D << "," << run_id << "," << t << "," << ms << "," << L << "\n";
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    RunResult rr;
    rr.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rr.per_iter_ms = rr.total_ms / iters;
    rr.final_loss = last_loss; return rr;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    Args A = parse_args(argc, argv);
#ifdef _OPENMP
    if (A.threads > 0)
        omp_set_num_threads(A.threads);
#endif
    if (A.n_samples <= 0 || A.n_features <= 0 || A.lr <= 0) {
        fprintf(stderr, "Invalid params\n");
        return 2;
    }
    if (A.pilot_iters <= 0)
        A.pilot_iters = 5;
    if (A.runs <= 0)
        A.runs = 1;

    Data d; generate_data(d, A.n_samples, A.n_features, A.seed);
    std::vector<double> w(A.n_features, 0.0);

    int threads =
#ifdef _OPENMP
    (A.threads > 0 ? A.threads : omp_get_max_threads());
#else
        1;
#endif

    int iters = A.iters;
    if (A.target_sec > 0) {
        std::vector<double> wp = w;
        int need = calibrate_iters(d, wp, A.lr, A.pilot_iters, A.target_sec);
        if (need > 0) {
            iters = need;
            std::fill(w.begin(), w.end(), 0.0);
            (void)compute_loss(d, w);
        }
    }

    ensure_header(A.csv_path, "date,lang,cpu_label,n_samples,n_features,iters,lr,threads,time_ms_total,time_ms_per_iter,loss_final,runs");

    double sum_total = 0.0, sum_piter = 0.0;
    double last_loss = 0.0;
    for (int run = 1; run <= A.runs; ++run) {
        std::fill(w.begin(), w.end(), 0.0);
        std::string per = A.per_iter_log;
        if (!per.empty()) {
            std::string base = per, ext = "";
            auto p = per.find_last_of('.');
            if (p != std::string::npos) {
                base = per.substr(0, p);
                ext = per.substr(p);
            }
            per = base + "_run" + std::to_string(run) + ext;
        }
        RunResult rr = run_once(d, w, A.lr, iters, A.label, threads, per, A.log_header_force, run);
        sum_total += rr.total_ms;
        sum_piter += rr.per_iter_ms;
        last_loss = rr.final_loss;
    }

    double avg_total = sum_total / A.runs, avg_piter = sum_piter / A.runs;
    std::ofstream ofs(A.csv_path, std::ios::app);
    ofs << now_str() << ",cpp," << A.label << "," << A.n_samples << "," << A.n_features << "," << iters << "," << A.lr << "," << threads << ","
        << avg_total << "," << avg_piter << "," << last_loss << "," << A.runs << "\n";

    std::cout << "Done. total_ms_avg=" << avg_total << " per_iter_ms_avg=" << avg_piter << " loss=" << last_loss
        << " threads=" << threads << " iters=" << iters << " runs=" << A.runs;
    if (A.target_sec > 0) std::cout << " (autocalibrated)";
    std::cout << std::endl;
    return 0;
}
