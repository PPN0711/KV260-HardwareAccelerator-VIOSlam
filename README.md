# KV260 Hardware Accelerator for VIO - Delta Trust Optimization

[![Target Device](https://img.shields.io/badge/Device-Kria%20KV260%20(xck26)-blue.svg)]()
[![Tool](https://img.shields.io/badge/Tool-Vitis%20HLS%202026.1-green.svg)]()
[![Status](https://img.shields.io/badge/Timing-PASS%20(WNS%200.408ns)-brightgreen.svg)]()
[![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20HLS%20%7C%20Verilog-orange.svg)]()

## 📖 Giới thiệu (Overview)

Dự án này tập trung vào việc **tăng tốc phần cứng (Hardware Acceleration)** cho thuật toán tối ưu hóa **Delta Trust** trong bài toán **VIO (Visual-Inertial Odometry)**. 

Trong các hệ thống SLAM/VIO, bước giải hệ phương trình phi tuyến (Non-linear Least Squares) thông qua phân rã ma trận Hessian (LDL^T) và phương pháp Dogleg là một trong những "nút thắt cổ chai" (bottleneck) lớn nhất về mặt thời gian tính toán trên CPU. Dự án này sử dụng **Vitis HLS** để chuyển đổi thuật toán C++ sang phần cứng FPGA, giúp tăng tốc độ xử lý và giảm độ trễ (latency).

*   **Target Board:** Kria KV260 Vision AI Starter Kit (Part: `xck26-sfvc784-2LV-c`)
*   **Số chiều ma trận tối đa (N):** 16 (Bị giới hạn bởi tài nguyên BRAM của FPGA)
*   **Kiểu dữ liệu:** `ap_fixed<32, 16>` (Số thực dấu phẩy tĩnh, 32-bit)

---

## 📊 Kết quả đạt được (Key Results)

Nhờ quá trình tối ưu hóa Pipeline và chia nhỏ mảng (Array Partitioning) trong HLS, thiết kế đã đạt được các thông số ấn tượng:

| Chỉ số | Kết quả | Ghi chú |
| :--- | :--- | :--- |
| **Timing (Vivado)** | **WNS = 0.408 ns**, TNS = 0.000 ns | Thiết kế hoàn toàn đạt timing ở tần số 100MHz |
| **Latency (C/RTL Co-sim)** | **3,605 clock cycles** | Tương đương **36.05 µs** ở tần số 100MHz |
| **Functionality** | **PASS** | Mô phỏng RTL khớp hoàn toàn với mô hình tham chiếu C++ |
| **Resource Utilization** | 1 BRAM, 78 DSP, ~14k FF, ~18k LUT | Tiết kiệm tài nguyên tối đa cho KV260 |

*(Ghi chú: Trong bài toán VIO thực tế với ma trận 270x270, CPU mất ~16.53ms/vòng lặp. Với thiết kế FPGA hiện tại (N=16), thời gian xử lý chỉ là 36.05 µs, cho thấy tiềm năng tăng tốc vượt trội nếu mở rộng bài toán).*

---

## 📂 Cấu trúc thư mục (Repository Structure)

```text
├── hls/                        # Thư mục chứa dự án Vitis HLS
│   ├── data/                   # Dữ liệu test (input.txt, golden.txt, q.txt...)
│   ├── src/                    # Mã nguồn C++ (HLS Core)
│   │   ├── top_hls.cpp         # Hàm top-level (delta_trust_top)
│   │   ├── delta_trust_hls.*   # Logic thuật toán Delta Trust
│   │   └── ldlt_hls.*          # Logic phân rã LDL^T và Dogleg
│   └── tb/                     # Testbench
│       ├── testbench.cpp       # Testbench chạy trên CPU (Benchmark)
│       └── testbench_top.cpp   # Testbench phục vụ C/RTL Co-simulation
├── src/                        # Mã nguồn C++ cho phần mềm (CPU - VIO SLAM)
│   ├── stage0/                 # Các hàm toán học cơ bản (LDLT, Dogleg)
│   ├── stage1/                 # Tiền xử lý IMU, Euroc Reader
│   ├── stage2/                 # Marginalization
│   └── stage3/                 # Tối ưu hóa và đánh giá ATE
├── Report/                     # Báo cáo tổng hợp HLS (.rpt)
│   ├── delta_trust_top_csynth.rpt
│   └── delta_trust_top_cosim.rpt
├── CMakeLists.txt              # File build hệ thống C++ phần mềm
├── build.sh                    # Script build tự động
└── README.md                   # Tài liệu dự án
