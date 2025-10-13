module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global private constant @".str"("MyPassword123\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @SECRET() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external @magic_value(-559038737 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>]
  llvm.func @validate(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb13
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %7 = llvm.load %6 {alignment = 1 : i64} : !llvm.ptr -> i8
    %8 = llvm.sext %7 : i8 to i32
    %9 = llvm.icmp "ne" %8, %1 : i32
    llvm.cond_br %9, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %10 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %11 = llvm.load %10 {alignment = 1 : i64} : !llvm.ptr -> i8
    %12 = llvm.sext %11 : i8 to i32
    %13 = llvm.icmp "ne" %12, %1 : i32
    llvm.br ^bb5(%13 : i1)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5(%2 : i1)
  ^bb5(%14: i1):  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    llvm.cond_br %14, ^bb7, ^bb14
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %15 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.load %15 {alignment = 1 : i64} : !llvm.ptr -> i8
    %17 = llvm.sext %16 : i8 to i32
    %18 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.load %18 {alignment = 1 : i64} : !llvm.ptr -> i8
    %20 = llvm.sext %19 : i8 to i32
    %21 = llvm.icmp "ne" %17, %20 : i32
    llvm.cond_br %21, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    llvm.store %1, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %22 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %22 : i32
  ^bb11:  // pred: ^bb9
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %23 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %24 = llvm.getelementptr %23[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %24, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %25 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.getelementptr %25[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %26, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb13
  ^bb13:  // pred: ^bb12
    llvm.br ^bb2
  ^bb14:  // pred: ^bb6
    llvm.br ^bb15
  ^bb15:  // pred: ^bb14
    %27 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.load %27 {alignment = 1 : i64} : !llvm.ptr -> i8
    %29 = llvm.sext %28 : i8 to i32
    %30 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.load %30 {alignment = 1 : i64} : !llvm.ptr -> i8
    %32 = llvm.sext %31 : i8 to i32
    %33 = llvm.icmp "eq" %29, %32 : i32
    %34 = llvm.zext %33 : i1 to i32
    llvm.store %34, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %35 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %35 : i32
  }
  llvm.func @process(%arg0: i32) -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.addressof @magic_value : !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %6 = llvm.xor %4, %5 : i32
    llvm.store %6, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %7 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %7 : i32
  }
  llvm.func @main() -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.addressof @SECRET : !llvm.ptr
    %2 = llvm.mlir.addressof @".str" : !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %5 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.call @validate(%2, %5) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.store %6, %4 {alignment = 4 : i64} : i32, !llvm.ptr
    %7 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %8 = llvm.call @process(%7) : (i32) -> i32
    llvm.store %8, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %9 : i32
  }
}
