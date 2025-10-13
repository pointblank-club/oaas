; ModuleID = '/home/zrahay/oaas/mlir_e2e_test_20251204_034403/simple_clangir_test.c'
source_filename = "/home/zrahay/oaas/mlir_e2e_test_20251204_034403/simple_clangir_test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private constant [14 x i8] c"MyPassword123\00", align 1
@SECRET = dso_local global ptr @.str, align 8
@magic_value = dso_local global i32 -559038737, align 4

; Function Attrs: noinline
define dso_local i32 @validate(ptr %0, ptr %1) #0 {
  %3 = alloca ptr, i64 1, align 8
  %4 = alloca ptr, i64 1, align 8
  %5 = alloca i32, i64 1, align 4
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  br label %6

6:                                                ; preds = %2
  br label %7

7:                                                ; preds = %39, %6
  %8 = load ptr, ptr %3, align 8
  %9 = load i8, ptr %8, align 1
  %10 = sext i8 %9 to i32
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %17

12:                                               ; preds = %7
  %13 = load ptr, ptr %4, align 8
  %14 = load i8, ptr %13, align 1
  %15 = sext i8 %14 to i32
  %16 = icmp ne i32 %15, 0
  br label %18

17:                                               ; preds = %7
  br label %18

18:                                               ; preds = %12, %17
  %19 = phi i1 [ false, %17 ], [ %16, %12 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %40

21:                                               ; preds = %20
  br label %22

22:                                               ; preds = %21
  br label %23

23:                                               ; preds = %22
  %24 = load ptr, ptr %3, align 8
  %25 = load i8, ptr %24, align 1
  %26 = sext i8 %25 to i32
  %27 = load ptr, ptr %4, align 8
  %28 = load i8, ptr %27, align 1
  %29 = sext i8 %28 to i32
  %30 = icmp ne i32 %26, %29
  br i1 %30, label %31, label %33

31:                                               ; preds = %23
  store i32 0, ptr %5, align 4
  %32 = load i32, ptr %5, align 4
  ret i32 %32

33:                                               ; preds = %23
  br label %34

34:                                               ; preds = %33
  %35 = load ptr, ptr %3, align 8
  %36 = getelementptr i8, ptr %35, i64 1
  store ptr %36, ptr %3, align 8
  %37 = load ptr, ptr %4, align 8
  %38 = getelementptr i8, ptr %37, i64 1
  store ptr %38, ptr %4, align 8
  br label %39

39:                                               ; preds = %34
  br label %7

40:                                               ; preds = %20
  br label %41

41:                                               ; preds = %40
  %42 = load ptr, ptr %3, align 8
  %43 = load i8, ptr %42, align 1
  %44 = sext i8 %43 to i32
  %45 = load ptr, ptr %4, align 8
  %46 = load i8, ptr %45, align 1
  %47 = sext i8 %46 to i32
  %48 = icmp eq i32 %44, %47
  %49 = zext i1 %48 to i32
  store i32 %49, ptr %5, align 4
  %50 = load i32, ptr %5, align 4
  ret i32 %50
}

; Function Attrs: noinline
define dso_local i32 @process(i32 %0) #0 {
  %2 = alloca i32, i64 1, align 4
  %3 = alloca i32, i64 1, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = load i32, ptr @magic_value, align 4
  %6 = xor i32 %4, %5
  store i32 %6, ptr %3, align 4
  %7 = load i32, ptr %3, align 4
  ret i32 %7
}

; Function Attrs: noinline
define dso_local i32 @main() #0 {
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  %3 = load ptr, ptr @SECRET, align 8
  %4 = call i32 @validate(ptr @.str, ptr %3)
  store i32 %4, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = call i32 @process(i32 %5)
  store i32 %6, ptr %1, align 4
  %7 = load i32, ptr %1, align 4
  ret i32 %7
}

attributes #0 = { noinline }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
