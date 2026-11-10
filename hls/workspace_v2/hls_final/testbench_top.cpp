#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>

#include "C:/msys64/home/shinr/delta_trust_vio/hls/src/delta_trust_hls.h"
using namespace std;

int main() {
    int n = 16;
    data_t S[MAX_N][MAX_N];
    data_t q[MAX_N];
    data_t step[MAX_N];
    data_t radius = 1.0;

    // Đọc dữ liệu từ file (giữ nguyên đường dẫn của bạn)
    ifstream fin("C:/msys64/home/shinr/delta_trust_vio/hls/data/input.txt");
    int n_file; fin >> n_file;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double val; fin >> val; S[i][j] = data_t(val);
        }
    
    ifstream qin("C:/msys64/home/shinr/delta_trust_vio/hls/data/q.txt");
    qin >> n_file;
    for (int i = 0; i < n; ++i) {
        double val; qin >> val; q[i] = data_t(val);
    }

    // GỌI HÀM TOP-LEVEL TẠI ĐÂY
    delta_trust_top(S, q, step, radius, n);

    cout << "Testbench hoan thanh!\n";
    return 0;
}