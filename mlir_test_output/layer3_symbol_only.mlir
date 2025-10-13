module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 22.0.0git (https://github.com/SkySingh04/llvm-project.git eba35723958cf6da53e6ac7c8223ce914047bca4)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global private unnamed_addr constant @".str"("SuperSecretPassword2024!\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @MASTER_PASSWORD() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str.1"("sk_live_abc123_secret_key_xyz789\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @API_KEY() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str.2"("postgresql://admin:secret@localhost:5432/db\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @DATABASE_URL() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external constant @MAGIC_NUMBER(-559038737 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external constant @ENCRYPTION_FACTOR(3.141590e+00 : f32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : f32
  llvm.mlir.global private unnamed_addr constant @".str.3"("Access GRANTED! Magic: %d, Encrypted: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"("Invalid API key\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("Invalid password\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.6"("=== MLIR Obfuscation Test ===\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("Testing authentication...\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("Authentication result: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"("wrong_password\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"("wrong_key\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.11"("Bad auth result: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.12"("Test complete!\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.func @f_61ae7c24(%arg0: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @MASTER_PASSWORD : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %4 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %5 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.call @strcmp(%4, %5) {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = read, targetMem0 = read, targetMem1 = read>, no_unwind, will_return} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    %7 = llvm.icmp "eq" %6, %2 : i32
    %8 = llvm.zext %7 : i1 to i32
    llvm.return %8 : i32
  }
  llvm.func @strcmp(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = read, targetMem0 = read, targetMem1 = read>, no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
  llvm.func @f_0218a827(%arg0: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @API_KEY : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %4 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %5 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.call @strcmp(%4, %5) {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = read, targetMem0 = read, targetMem1 = read>, no_unwind, will_return} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    %7 = llvm.icmp "eq" %6, %2 : i32
    %8 = llvm.zext %7 : i1 to i32
    llvm.return %8 : i32
  }
  llvm.func @f_f8b7e158(%arg0: i32 {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(-559038737 : i32) : i32
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.xor %3, %1 : i32
    llvm.return %4 : i32
  }
  llvm.func @f_b9127cd9(%arg0: f32 {llvm.noundef}) -> f32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.141590e+00 : f32) : f32
    %2 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : f32, !llvm.ptr
    %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> f32
    %4 = llvm.fmul %3, %1 : f32
    llvm.return %4 : f32
  }
  llvm.func @f_b610cf36(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %4 = llvm.mlir.constant(42 : i32) : i32
    %5 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %11 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.call @f_61ae7c24(%11) : (!llvm.ptr {llvm.noundef}) -> i32
    %13 = llvm.icmp "ne" %12, %1 : i32
    llvm.cond_br %13, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %14 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.call @f_0218a827(%14) : (!llvm.ptr {llvm.noundef}) -> i32
    %16 = llvm.icmp "ne" %15, %1 : i32
    llvm.cond_br %16, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %17 = llvm.call @f_f8b7e158(%4) : (i32 {llvm.noundef}) -> i32
    llvm.store %17, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    %18 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.sitofp %18 : i32 to f32
    %20 = llvm.call @f_b9127cd9(%19) : (f32 {llvm.noundef}) -> f32
    llvm.store %20, %10 {alignment = 4 : i64} : f32, !llvm.ptr
    %21 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %22 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> f32
    %23 = llvm.fpext %22 : f32 to f64
    %24 = llvm.call @printf(%5, %21, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    llvm.store %0, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb3:  // pred: ^bb1
    %25 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb0
    %26 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 3 preds: ^bb2, ^bb3, ^bb4
    %27 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %27 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func @main() -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %6 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %7 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %8 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %9 = llvm.mlir.addressof @".str.11" : !llvm.ptr
    %10 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %11 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    %14 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    %15 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    %16 = llvm.call @f_b610cf36(%4, %5) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.store %16, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    %17 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.call @printf(%6, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %19 = llvm.call @f_b610cf36(%7, %8) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.store %19, %13 {alignment = 4 : i64} : i32, !llvm.ptr
    %20 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.call @printf(%9, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %22 = llvm.call @printf(%10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.return %1 : i32
  }
}

