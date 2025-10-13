; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [15 x i8] c"TopSecret2024!\00", align 1
@MASTER_PASSWORD = dso_local global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [22 x i8] c"sk_live_xyz123_secret\00", align 1
@API_KEY = dso_local global ptr @.str.1, align 8
@.str.2 = private unnamed_addr constant [34 x i8] c"postgres://admin:password@db/main\00", align 1
@DB_CONN = dso_local global ptr @.str.2, align 8
@MAGIC_NUMBER = dso_local constant i32 -559038737, align 4
@PI_FACTOR = dso_local constant float 0x400921FA00000000, align 4
@.str.3 = private unnamed_addr constant [28 x i8] c"ACCESS GRANTED! Result: %f\0A\00", align 1
@.str.4 = private unnamed_addr constant [18 x i8] c"Invalid API key!\0A\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"Invalid password!\0A\00", align 1
@.str.6 = private unnamed_addr constant [23 x i8] c"=== OAAS E2E Test ===\0A\00", align 1
@.str.7 = private unnamed_addr constant [20 x i8] c"Test 1 (valid): %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [6 x i8] c"wrong\00", align 1
@.str.9 = private unnamed_addr constant [22 x i8] c"Test 2 (invalid): %d\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_61ae7c24(ptr noundef %0) #0 {
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
define dso_local i32 @f_0218a827(ptr noundef %0) #0 {
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
define dso_local i32 @f_f8b7e158(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = xor i32 %3, -559038737
  ret i32 %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local float @f_b9127cd9(float noundef %0) #0 {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = fmul float %3, 0x400921FA00000000
  ret float %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_b610cf36(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca float, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call i32 @f_61ae7c24(ptr noundef %8)
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %25

11:                                               ; preds = %2
  %12 = load ptr, ptr %5, align 8
  %13 = call i32 @f_0218a827(ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %23

15:                                               ; preds = %11
  %16 = call i32 @f_f8b7e158(i32 noundef 42)
  store i32 %16, ptr %6, align 4
  %17 = load i32, ptr %6, align 4
  %18 = sitofp i32 %17 to float
  %19 = call float @f_b9127cd9(float noundef %18)
  store float %19, ptr %7, align 4
  %20 = load float, ptr %7, align 4
  %21 = fpext float %20 to double
  %22 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, double noundef %21)
  store i32 1, ptr %3, align 4
  br label %27

23:                                               ; preds = %11
  %24 = call i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i32 0, ptr %3, align 4
  br label %27

25:                                               ; preds = %2
  %26 = call i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i32 0, ptr %3, align 4
  br label %27

27:                                               ; preds = %15, %23, %25
  %28 = load i32, ptr %3, align 4
  ret i32 %28
}

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 %0, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  %8 = call i32 (ptr, ...) @printf(ptr noundef @.str.6)
  %9 = call i32 @f_b610cf36(ptr noundef @.str, ptr noundef @.str.1)
  store i32 %9, ptr %6, align 4
  %10 = load i32, ptr %6, align 4
  %11 = call i32 (ptr, ...) @printf(ptr noundef @.str.7, i32 noundef %10)
  %12 = call i32 @f_b610cf36(ptr noundef @.str.8, ptr noundef @.str.8)
  store i32 %12, ptr %7, align 4
  %13 = load i32, ptr %7, align 4
  %14 = call i32 (ptr, ...) @printf(ptr noundef @.str.9, i32 noundef %13)
  %15 = load i32, ptr %6, align 4
  %16 = icmp eq i32 %15, 1
  br i1 %16, label %17, label %20

17:                                               ; preds = %2
  %18 = load i32, ptr %7, align 4
  %19 = icmp eq i32 %18, 0
  br label %20

20:                                               ; preds = %17, %2
  %21 = phi i1 [ %19, %17 ], [ false, %2 ]
  %22 = zext i1 %21 to i64
  %23 = select i1 %21, i32 0, i32 1
  ret i32 %23
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #3 = { nounwind willreturn memory(read) }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
