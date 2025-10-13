; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
target triple = "]P>CXY/HX*"

@.str = private unnamed_addr constant [30 x i8] c",\00\0A\0D\1AL\12-\04\08Y\0B\07\00\14\06\0F\15+\0E\01Y\06\0C\08\00\07\15U_", align 1
@secret_message = dso_local global ptr @.str, align 8
@magic_number = dso_local constant i32 42, align 4
@.str.1 = private unnamed_addr constant [13 x i8] c")\00\15\12\14\0B\11eK@\0Ane", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c")\04\01\08\16VTz\0Foy", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"7\10\0B[UI\10Uk", align 1
@.str.4 = private unnamed_addr constant [13 x i8] c"4\17\09\05\00\0F\00eK@\1Dne", align 1

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

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="\1C]PLCX" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="\03\00\08\04\07\05\17" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"1\07\13\0F\01\19T<\07\04\17\03E\10\04\07\1F\1D0\05EH\\KWOFL\\n\1E\07\0C\0A\11\13P\\"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
