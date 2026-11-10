#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <chrono> // Thư viện đo thời gian
#include "../src/delta_trust_hls.h"
using namespace std;

static double max_LD_diff(
    data_t L1[MAX_N][MAX_N], data_t D1[MAX_N],
    data_t L2[MAX_N][MAX_N], data_t D2[MAX_N],
    int n)
{
    double m = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m = max(m, fabs((double)L1[i][j] - (double)L2[i][j]));
    for (int i = 0; i < n; ++i)
        m = max(m, fabs((double)D1[i] - (double)D2[i]));
    return m;
}

int main() {

    printf("\n=== BAT DAU CHAY TESTBENCH ===\n");
    fflush(stdout);

    const int n = 16;
    data_t S[MAX_N][MAX_N];
    data_t L_gold[MAX_N][MAX_N], D_gold[MAX_N];
    data_t q[MAX_N];
    data_t step_gold[MAX_N];
    data_t v[MAX_N];
    data_t sigma;
    data_t L_upd_gold[MAX_N][MAX_N], D_upd_gold[MAX_N];
    string dogleg_kind;

    // --- ĐỌC DỮ LIỆU (Giữ nguyên) ---
    ifstream fin("C:/msys64/home/shinr/delta_trust_vio/hls/data/input.txt");
    if (!fin.is_open()) { printf("KHONG MO DUOC data/input.txt!\n"); return 1; }
    int n_file; fin >> n_file;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) { double val; fin >> val; S[i][j] = data_t(val); }

    ifstream gin("C:/msys64/home/shinr/delta_trust_vio/hls/data/golden.txt");
    gin >> n_file;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) { double val; gin >> val; L_gold[i][j] = data_t(val); }
    for (int i = 0; i < n; ++i) { double val; gin >> val; D_gold[i] = data_t(val); }

    ifstream qin("C:/msys64/home/shinr/delta_trust_vio/hls/data/q.txt");
    qin >> n_file;
    for (int i = 0; i < n; ++i) { double val; qin >> val; q[i] = data_t(val); }

    ifstream dgin("C:/msys64/home/shinr/delta_trust_vio/hls/data/dogleg_golden.txt");
    dgin >> n_file >> dogleg_kind;
    for (int i = 0; i < n; ++i) { double val; dgin >> val; step_gold[i] = data_t(val); }

    ifstream vin("C:/msys64/home/shinr/delta_trust_vio/hls/data/rank1_v.txt");
    vin >> n_file >> sigma;
    for (int i = 0; i < n; ++i) { double val; vin >> val; v[i] = data_t(val); }

    ifstream uin("C:/msys64/home/shinr/delta_trust_vio/hls/data/rank1_golden.txt");
    int ok_gold; uin >> n_file >> ok_gold;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) { double val; uin >> val; L_upd_gold[i][j] = data_t(val); }
    for (int i = 0; i < n; ++i) { double val; uin >> val; D_upd_gold[i] = data_t(val); }

    // ============ KIỂM TRA TÍNH ĐÚNG ĐẮN ============
    data_t L_hls[MAX_N][MAX_N], D_hls[MAX_N];
    data_t step[MAX_N];
    data_t L_upd[MAX_N][MAX_N], D_upd[MAX_N];

    ldlt_decompose_hls(S, L_hls, D_hls, n);
    dogleg_hls(L_hls, D_hls, q, step, data_t(1.0), n);
    
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            L_upd[i][j] = L_hls[i][j];
    for (int i = 0; i < n; ++i) D_upd[i] = D_hls[i];
    ldlt_rank1_update_hls(L_upd, D_upd, v, sigma, n);

    double err_ld = max_LD_diff(L_hls, D_hls, L_gold, D_gold, n);
    double err_dl = 0;
    for (int i = 0; i < n; ++i) err_dl = max(err_dl, fabs((double)step[i] - (double)step_gold[i]));
    double err_upd = max_LD_diff(L_upd, D_upd, L_upd_gold, D_upd_gold, n);
    
    bool pass_all = (err_ld < 1e-3) && (err_dl < 1e-3) && (err_upd < 1e-3);
    cout << "=== KET QUA KIEM TRA ===\n";
    cout << "  " << (pass_all ? "[PASS] Tat ca deu dung!" : "[FAIL] Co loi xay ra!") << "\n\n";

    // ============ ĐO THỜI GIAN CHẠY TRÊN CPU ============
    cout << "=== BENCHMARK CPU (n = " << n << ") ===\n";
    const int ITERATIONS = 10000; // Chạy 10,000 lần để lấy thời gian trung bình
    cout << "Dang chay " << ITERATIONS << " lan de do thoi gian...\n";

    // Bắt đầu bấm đồng hồ
    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Chạy lại toàn bộ luồng tính toán giống hệt hàm top delta_trust_top
        ldlt_decompose_hls(S, L_hls, D_hls, n);
        dogleg_hls(L_hls, D_hls, q, step, data_t(1.0), n);
        
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                L_upd[i][j] = L_hls[i][j];
        for (int i = 0; i < n; ++i) D_upd[i] = D_hls[i];
        ldlt_rank1_update_hls(L_upd, D_upd, v, sigma, n);
    }

    // Dừng đồng hồ
    auto end = chrono::high_resolution_clock::now();

    // Tính toán thời gian
    chrono::duration<double, std::micro> elapsed = end - start;
    double total_time_us = elapsed.count();
    double time_per_run_us = total_time_us / ITERATIONS;

    cout << fixed << setprecision(2);
    cout << "  Tong thoi gian: " << total_time_us << " us\n";
    cout << "  Thoi gian 1 lan chay: " << time_per_run_us << " us\n";
    cout << "  So sanh voi FPGA (36.05 us): ";
    
    if (time_per_run_us > 36.05) {
        cout << "FPGA nhanh hon " << time_per_run_us / 36.05 << " lan\n";
    } else {
        cout << "CPU nhanh hon " << 36.05 / time_per_run_us << " lan\n";
    }
    cout << "========================================\n";

    return 0;
}