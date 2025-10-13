; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
target triple = "x86_64-unknown-linux-gnu"
"

@.str = private unnamed_addr constant [25 x i8] c"7\10\16\04\07?\11<\19\00\0D4\04\15\12\02\03\06;YUKPDf", align 1
@MASTER_PASSWORD = dso_local global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [33 x i8] c"\17\0E9\0D\1C\1A\11\00\0A\07\1AUWU>\06\09\17-\0E\11&\0F\00\1F>\0D\15\0EhS\\y", align 1
@API_KEY = dso_local global ptr @.str.1, align 8
@.str.2 = private unnamed_addr constant [44 x i8] c"\14\0A\15\15\12\1E\11,\1A\09CKJ\07\05\18\05\1Ae\18\00\1A\16\00\12!\19\03\17>\07\0D\16\17\11\\TA_Fp\0F\07y", align 1
@DATABASE_URL = dso_local global ptr @.str.2, align 8
@MAGIC_NUMBER = dso_local constant i32 -559038737, align 4
@ENCRYPTION_FACTOR = dso_local constant float 0x400921FA00000000, align 4
@.str.3 = private unnamed_addr constant [42 x i8] c"%\06\05\04\06\1FT\189$70 \22@U!\158\02\06CD@\02MU)\1A<\19\1C\09\10\00\02[UI\12Uk", align 1
@.str.4 = private unnamed_addr constant [17 x i8] c"-\0B\10\00\19\05\10\7F*50D\0E\03\18\7Fl", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"-\0B\10\00\19\05\10\7F\1B\04\0A\17\12\09\13\11ft", align 1
@.str.6 = private unnamed_addr constant [31 x i8] c"YX[A8 =\0DK*\1B\02\10\15\02\14\18\1D0\05E-\01\16\12AHQIUk", align 1
@.str.7 = private unnamed_addr constant [27 x i8] c"0\00\15\15\1C\02\13\7F\0A\10\0D\0C\00\08\15\1C\0F\15+\02\0A\17JKHku", align 1
@.str.8 = private unnamed_addr constant [27 x i8] c"%\10\12\09\10\02\006\08\04\0D\0D\0A\08A\07\09\07*\07\11CD@\02ku", align 1
@.str.9 = private unnamed_addr constant [15 x i8] c"\13\17\09\0F\123\04>\18\16\0E\0B\17\02a", align 1
@.str.10 = private unnamed_addr constant [10 x i8] c"\13\17\09\0F\123\1F:\12e", align 1
@.str.11 = private unnamed_addr constant [21 x i8] c"&\04\02A\14\19\007K\17\1C\17\10\0A\15OLQ;ae", align 1
@.str.12 = private unnamed_addr constant [16 x i8] c"0\00\15\15U\0F\1B2\1B\09\1C\10\00Gku", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_dabe0a778dd2(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr @MASTER_PASSWORD, align 8
  %5 = call i32 @f_64205d6e1d5e(ptr noundef %3, ptr noundef %4) #3
  %6 = icmp eq i32 %5, 0
  %7 = zext i1 %6 to i32
  ret i32 %7
}

; Function Attrs: nounwind willreturn memory(read)
declare i32 @f_64205d6e1d5e(ptr noundef, ptr noundef) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_efa7967eca55(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr @API_KEY, align 8
  %5 = call i32 @f_64205d6e1d5e(ptr noundef %3, ptr noundef %4) #3
  %6 = icmp eq i32 %5, 0
  %7 = zext i1 %6 to i32
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_83792c93bd20(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = xor i32 %3, -559038737
  ret i32 %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local float @f_98d3061f8534(float noundef %0) #0 {
  %2 = alloca float, align 4
  store float %0, ptr %2, align 4
  %3 = load float, ptr %2, align 4
  %4 = fmul float %3, 0x400921FA00000000
  ret float %4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_0a9fc93cc940(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca float, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call i32 @f_dabe0a778dd2(ptr noundef %8)
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %26

11:                                               ; preds = %2
  %12 = load ptr, ptr %5, align 8
  %13 = call i32 @f_efa7967eca55(ptr noundef %12)
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %24

15:                                               ; preds = %11
  %16 = call i32 @f_83792c93bd20(i32 noundef 42)
  store i32 %16, ptr %6, align 4
  %17 = load i32, ptr %6, align 4
  %18 = sitofp i32 %17 to float
  %19 = call float @f_98d3061f8534(float noundef %18)
  store float %19, ptr %7, align 4
  %20 = load i32, ptr %6, align 4
  %21 = load float, ptr %7, align 4
  %22 = fpext float %21 to double
  %23 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.3, i32 noundef %20, double noundef %22)
  store i32 1, ptr %3, align 4
  br label %28

24:                                               ; preds = %11
  %25 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.4)
  store i32 0, ptr %3, align 4
  br label %28

26:                                               ; preds = %2
  %27 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.5)
  store i32 0, ptr %3, align 4
  br label %28

28:                                               ; preds = %15, %24, %26
  %29 = load i32, ptr %3, align 4
  ret i32 %29
}

declare i32 @f_2abfca9157e9(ptr noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %4 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.6)
  %5 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.7)
  %6 = call i32 @f_0a9fc93cc940(ptr noundef @.str, ptr noundef @.str.1)
  store i32 %6, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  %8 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.8, i32 noundef %7)
  %9 = call i32 @f_0a9fc93cc940(ptr noundef @.str.9, ptr noundef @.str.10)
  store i32 %9, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  %11 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.11, i32 noundef %10)
  %12 = call i32 (ptr, ...) @f_2abfca9157e9(ptr noundef @.str.12)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="\1C]PLCX" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="\03\00\08\04\07\05\17" }
attributes #1 = { nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #3 = { nounwind willreturn memory(read) }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"\07\09\07\0F\12L\02:\19\16\10\0B\0BFSGBDq[\02\10\10EN\09\01\18\04,QJV\03\0C\12\09\00\0EZ<\04\08V7\0E\1F2\1C\02\137[QV\08\09\10\0CX\1C\060\01\00\1A\10K\01\08\01L\11=\0AVLSWUX@T\179]\01\18QV\03W\14\0FC<SWKW\06\03XDXDk\\\07\1A\05QO"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
