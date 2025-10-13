module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "Ubuntu clang version 18.1.3 (1ubuntu1)", llvm.module_asm = [], llvm.target_triple = "x86_64-pc-linux-gnu"} {
  llvm.mlir.global private unnamed_addr constant @".str"("Hello from obfuscated binary!\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @secret_message() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external constant @magic_number(42 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global private unnamed_addr constant @".str.1"("Message: %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.2"("Magic: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.3"("Sum: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"("Product: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.func @add_numbers(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.add %3, %4 overflow<nsw> : i32
    llvm.return %5 : i32
  }
  llvm.func @multiply(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.mul %3, %4 overflow<nsw> : i32
    llvm.return %5 : i32
  }
  llvm.func @main() -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @secret_message : !llvm.ptr
    %3 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %4 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %5 = llvm.mlir.constant(42 : i32) : i32
    %6 = llvm.mlir.constant(10 : i32) : i32
    %7 = llvm.mlir.constant(20 : i32) : i32
    %8 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %9 = llvm.mlir.constant(6 : i32) : i32
    %10 = llvm.mlir.constant(7 : i32) : i32
    %11 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    %13 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %14 = llvm.call @printf(%3, %13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    %15 = llvm.call @printf(%4, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %16 = llvm.call @add_numbers(%6, %7) : (i32 {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %17 = llvm.call @printf(%8, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %18 = llvm.call @multiply(%9, %10) : (i32 {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %19 = llvm.call @printf(%11, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    llvm.return %1 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
}
