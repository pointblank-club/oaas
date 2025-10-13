module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "\07\09\07\0F\12L\02:\19\16\10\0B\0BFSGBDq[\02\10\10EN\09\01\18\04,QJV\03\0C\12\09\00\0EZ<\04\08V7\0E\1F2\1C\02\137[QV\08\09\10\0CX\1C\060\01\00\1A\10K\01\08\01L\11=\0AVLSWUX@T\179]\01\18QV\03W\14\0FC<SWKW\06\03XDXDk\\\07\1A\05QO", llvm.module_asm = [], llvm.target_triple = "\1C]P>CXY*\05\0E\17\0B\12\08L\19\05\1A*\13H\1E\0A\10"} {
  llvm.mlir.global private unnamed_addr constant @".str"("7\10\16\04\07?\11<\19\00\0D4\04\15\12\02\03\06;YUKPDf") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @MASTER_PASSWORD() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str.1"("\17\0E9\0D\1C\1A\11\00\0A\07\1AUWU>\06\09\17-\0E\11&\0F\00\1F>\0D\15\0EhS\\y") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @API_KEY() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str.2"("\14\0A\15\15\12\1E\11,\1A\09CKJ\07\05\18\05\1Ae\18\00\1A\16\00\12!\19\03\17>\07\0D\16\17\11\\TA_Fp\0F\07y") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @DATABASE_URL() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external constant @MAGIC_NUMBER(-559038737 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external constant @ENCRYPTION_FACTOR(3.141590e+00 : f32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : f32
  llvm.mlir.global private unnamed_addr constant @".str.3"("%\06\05\04\06\1FT\189$70 \22@U!\158\02\06CD@\02MU)\1A<\19\1C\09\10\00\02[UI\12Uk") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"("-\0B\10\00\19\05\10\7F*50D\0E\03\18\7Fl") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("-\0B\10\00\19\05\10\7F\1B\04\0A\17\12\09\13\11ft") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.6"("YX[A8 =\0DK*\1B\02\10\15\02\14\18\1D0\05E-\01\16\12AHQIUk") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("0\00\15\15\1C\02\13\7F\0A\10\0D\0C\00\08\15\1C\0F\15+\02\0A\17JKHku") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("%\10\12\09\10\02\006\08\04\0D\0D\0A\08A\07\09\07*\07\11CD@\02ku") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"("\13\17\09\0F\123\04>\18\16\0E\0B\17\02a") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"("\13\17\09\0F\123\1F:\12e") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.11"("&\04\02A\14\19\007K\17\1C\17\10\0A\15OLQ;ae") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.12"("0\00\15\15U\0F\1B2\1B\09\1C\10\00Gku") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.func @validate_password(%arg0: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
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
  llvm.func @strcmp(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = read, targetMem0 = read, targetMem1 = read>, no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", will_return}
  llvm.func @check_api_key(%arg0: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
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
  llvm.func @process_magic(%arg0: i32 {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(-559038737 : i32) : i32
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = llvm.xor %3, %1 : i32
    llvm.return %4 : i32
  }
  llvm.func @compute_encrypted_value(%arg0: f32 {llvm.noundef}) -> f32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(3.141590e+00 : f32) : f32
    %2 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : f32, !llvm.ptr
    %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> f32
    %4 = llvm.fmul %3, %1 : f32
    llvm.return %4 : f32
  }
  llvm.func @authenticate_user(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
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
    %12 = llvm.call @validate_password(%11) : (!llvm.ptr {llvm.noundef}) -> i32
    %13 = llvm.icmp "ne" %12, %1 : i32
    llvm.cond_br %13, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %14 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.call @check_api_key(%14) : (!llvm.ptr {llvm.noundef}) -> i32
    %16 = llvm.icmp "ne" %15, %1 : i32
    llvm.cond_br %16, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %17 = llvm.call @process_magic(%4) : (i32 {llvm.noundef}) -> i32
    llvm.store %17, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    %18 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.sitofp %18 : i32 to f32
    %20 = llvm.call @compute_encrypted_value(%19) : (f32 {llvm.noundef}) -> f32
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
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17"}
  llvm.func @main() -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "\1C]PLCX", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "\03\00\08\04\07\05\17", uwtable_kind = #llvm.uwtableKind<async>} {
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
    %16 = llvm.call @authenticate_user(%4, %5) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.store %16, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    %17 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.call @printf(%6, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %19 = llvm.call @authenticate_user(%7, %8) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.store %19, %13 {alignment = 4 : i64} : i32, !llvm.ptr
    %20 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.call @printf(%9, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %22 = llvm.call @printf(%10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.return %1 : i32
  }
}

