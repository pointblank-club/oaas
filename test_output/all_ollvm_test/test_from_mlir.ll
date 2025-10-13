; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [30 x i8] c"Hello from obfuscated binary!\00", align 1
@secret_message = dso_local global ptr @.str, align 8
@magic_number = dso_local constant i32 42, align 4
@.str.1 = private unnamed_addr constant [13 x i8] c"Message: %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"Magic: %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"Sum: %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [13 x i8] c"Product: %d\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_61ae7c24(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_0218a827(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = mul nsw i32 %5, %6
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %2 = load ptr, ptr @secret_message, align 8
  %3 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, ptr noundef %2)
  %4 = call i32 (ptr, ...) @printf(ptr noundef @.str.2, i32 noundef 42)
  %5 = call i32 @f_61ae7c24(i32 noundef 10, i32 noundef 20)
  %6 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i32 noundef %5)
  %7 = call i32 @f_0218a827(i32 noundef 6, i32 noundef 7)
  %8 = call i32 (ptr, ...) @printf(ptr noundef @.str.4, i32 noundef %7)
  ret i32 0
}

declare i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8"    }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8"   }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
