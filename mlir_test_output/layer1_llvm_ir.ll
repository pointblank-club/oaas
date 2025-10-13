; ModuleID = 'mlir_test_output/test_secrets.c'
source_filename = "mlir_test_output/test_secrets.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [25 x i8] c"SuperSecretPassword2024!\00", align 1
@MASTER_PASSWORD = dso_local global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [33 x i8] c"sk_live_abc123_secret_key_xyz789\00", align 1
@API_KEY = dso_local global ptr @.str.1, align 8
@.str.2 = private unnamed_addr constant [44 x i8] c"postgresql://admin:secret@localhost:5432/db\00", align 1
@DATABASE_URL = dso_local global ptr @.str.2, align 8
@MAGIC_NUMBER = dso_local constant i32 -559038737, align 4
@ENCRYPTION_FACTOR = dso_local constant float 0x400921FA00000000, align 4
@.str.3 = private unnamed_addr constant [42 x i8] c"Access GRANTED! Magic: %d, Encrypted: %f\0A\00", align 1
@.str.4 = private unnamed_addr constant [17 x i8] c"Invalid API key\0A\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"Invalid password\0A\00", align 1
@.str.6 = private unnamed_addr constant [31 x i8] c"=== MLIR Obfuscation Test ===\0A\00", align 1
@.str.7 = private unnamed_addr constant [27 x i8] c"Testing authentication...\0A\00", align 1
@.str.8 = private unnamed_addr constant [27 x i8] c"Authentication result: %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [15 x i8] c"wrong_password\00", align 1
@.str.10 = private unnamed_addr constant [10 x i8] c"wrong_key\00", align 1
@.str.11 = private unnamed_addr constant [21 x i8] c"Bad auth result: %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [16 x i8] c"Test complete!\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @validate_password(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr @MASTER_PASSWORD, align 8
  %5 = call i32 @strcmp(ptr noundef %3, ptr noundef %4) #3
  %6 = icmp eq i32 %5, 0
  %7 = zext i1 %6 to i32
  ret i32 %7
}

; Function Attrs: nounwind willreturn memory(read)
declare i32 @strcmp(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @check_api_key(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr @API_KEY, align 8
  %5 = call i32 @strcmp(ptr noundef %3, ptr noundef %4) #3
  %6 = icmp eq i32 %5, 0
  %7 = zext i1 %6 to i32
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @process_magic(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = xor i32 %3, -559038737
  ret i32 %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local float @compute_encrypted_value(float noundef %0) #0 {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = fmul float %3, 0x400921FA00000000
  ret float %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @authenticate_user(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca float, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call i32 @validate_password(ptr noundef %8)
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %26

11:                                               ; preds = %2
  %12 = load ptr, ptr %5, align 8
  %13 = call i32 @check_api_key(ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %24

15:                                               ; preds = %11
  %16 = call i32 @process_magic(i32 noundef 42)
  store i32 %16, ptr %6, align 4
  %17 = load i32, ptr %6, align 4
  %18 = sitofp i32 %17 to float
  %19 = call float @compute_encrypted_value(float noundef %18)
  store float %19, ptr %7, align 4
  %20 = load i32, ptr %6, align 4
  %21 = load float, ptr %7, align 4
  %22 = fpext float %21 to double
  %23 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i32 noundef %20, double noundef %22)
  store i32 1, ptr %3, align 4
  br label %28

24:                                               ; preds = %11
  %25 = call i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i32 0, ptr %3, align 4
  br label %28

26:                                               ; preds = %2
  %27 = call i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i32 0, ptr %3, align 4
  br label %28

28:                                               ; preds = %26, %24, %15
  %29 = load i32, ptr %3, align 4
  ret i32 %29
}

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %4 = call i32 (ptr, ...) @printf(ptr noundef @.str.6)
  %5 = call i32 (ptr, ...) @printf(ptr noundef @.str.7)
  %6 = call i32 @authenticate_user(ptr noundef @.str, ptr noundef @.str.1)
  store i32 %6, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  %8 = call i32 (ptr, ...) @printf(ptr noundef @.str.8, i32 noundef %7)
  %9 = call i32 @authenticate_user(ptr noundef @.str.9, ptr noundef @.str.10)
  store i32 %9, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  %11 = call i32 (ptr, ...) @printf(ptr noundef @.str.11, i32 noundef %10)
  %12 = call i32 (ptr, ...) @printf(ptr noundef @.str.12)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 22.0.0git (https://github.com/SkySingh04/llvm-project.git eba35723958cf6da53e6ac7c8223ce914047bca4)"}
