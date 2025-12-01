; ModuleID = '/app/reports/5fceff438ada4b68a2e656946e3fa732/pasted_source_string_encrypted.c'
source_filename = "/app/reports/5fceff438ada4b68a2e656946e3fa732/pasted_source_string_encrypted.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [58 x i8] c"Hello from a 300-line C program with no headers! Value = \00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i64 @sys_write(i64 noundef %0, ptr noundef %1, i64 noundef %2) #0 {
  %4 = alloca i64, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  store i64 %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  %8 = load i64, ptr %4, align 8
  %9 = load ptr, ptr %5, align 8
  %10 = load i64, ptr %6, align 8
  %11 = call i64 asm sideeffect "mov $$1, %rax;\0Asyscall;\0A", "={ax},{di},{si},{dx},~{dirflag},~{fpsr},~{flags}"(i64 %8, ptr %9, i64 %10) #1, !srcloc !6
  store i64 %11, ptr %7, align 8
  %12 = load i64, ptr %7, align 8
  ret i64 %12
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @print_char(i8 noundef signext %0) #0 {
  %2 = alloca i8, align 1
  store i8 %0, ptr %2, align 1
  %3 = call i64 @sys_write(i64 noundef 1, ptr noundef %2, i64 noundef 1)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @print_str(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store ptr %0, ptr %2, align 8
  store i64 0, ptr %3, align 8
  br label %4

4:                                                ; preds = %10, %1
  %5 = load ptr, ptr %2, align 8
  %6 = load i64, ptr %3, align 8
  %7 = getelementptr inbounds i8, ptr %5, i64 %6
  %8 = load i8, ptr %7, align 1
  %9 = icmp ne i8 %8, 0
  br i1 %9, label %10, label %13

10:                                               ; preds = %4
  %11 = load i64, ptr %3, align 8
  %12 = add nsw i64 %11, 1
  store i64 %12, ptr %3, align 8
  br label %4, !llvm.loop !7

13:                                               ; preds = %4
  %14 = load ptr, ptr %2, align 8
  %15 = load i64, ptr %3, align 8
  %16 = call i64 @sys_write(i64 noundef 1, ptr noundef %14, i64 noundef %15)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @print_num_rec(i64 noundef %0) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i8, align 1
  store i64 %0, ptr %2, align 8
  %4 = load i64, ptr %2, align 8
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %1
  br label %15

7:                                                ; preds = %1
  %8 = load i64, ptr %2, align 8
  %9 = sdiv i64 %8, 10
  call void @print_num_rec(i64 noundef %9)
  %10 = load i64, ptr %2, align 8
  %11 = srem i64 %10, 10
  %12 = add nsw i64 48, %11
  %13 = trunc i64 %12 to i8
  store i8 %13, ptr %3, align 1
  %14 = load i8, ptr %3, align 1
  call void @print_char(i8 noundef signext %14)
  br label %15

15:                                               ; preds = %7, %6
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @print_num(i64 noundef %0) #0 {
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  call void @print_char(i8 noundef signext 48)
  br label %14

6:                                                ; preds = %1
  %7 = load i64, ptr %2, align 8
  %8 = icmp slt i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %6
  call void @print_char(i8 noundef signext 45)
  %10 = load i64, ptr %2, align 8
  %11 = sub nsw i64 0, %10
  store i64 %11, ptr %2, align 8
  br label %12

12:                                               ; preds = %9, %6
  %13 = load i64, ptr %2, align 8
  call void @print_num_rec(i64 noundef %13)
  br label %14

14:                                               ; preds = %12, %5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @newline() #0 {
  call void @print_char(i8 noundef signext 10)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f1() #0 {
  ret i32 1
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f2() #0 {
  ret i32 2
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f3() #0 {
  ret i32 3
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f4() #0 {
  ret i32 4
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f5() #0 {
  ret i32 5
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f6() #0 {
  ret i32 6
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f7() #0 {
  ret i32 7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f8() #0 {
  ret i32 8
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f9() #0 {
  ret i32 9
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f10() #0 {
  ret i32 10
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f11() #0 {
  ret i32 11
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f12() #0 {
  ret i32 12
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f13() #0 {
  ret i32 13
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f14() #0 {
  ret i32 14
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f15() #0 {
  ret i32 15
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f16() #0 {
  ret i32 16
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f17() #0 {
  ret i32 17
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f18() #0 {
  ret i32 18
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f19() #0 {
  ret i32 19
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f20() #0 {
  ret i32 20
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f21() #0 {
  ret i32 21
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f22() #0 {
  ret i32 22
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f23() #0 {
  ret i32 23
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f24() #0 {
  ret i32 24
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f25() #0 {
  ret i32 25
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f26() #0 {
  ret i32 26
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f27() #0 {
  ret i32 27
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f28() #0 {
  ret i32 28
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f29() #0 {
  ret i32 29
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f30() #0 {
  ret i32 30
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f31() #0 {
  ret i32 31
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f32() #0 {
  ret i32 32
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f33() #0 {
  ret i32 33
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f34() #0 {
  ret i32 34
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f35() #0 {
  ret i32 35
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f36() #0 {
  ret i32 36
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f37() #0 {
  ret i32 37
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f38() #0 {
  ret i32 38
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f39() #0 {
  ret i32 39
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f40() #0 {
  ret i32 40
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f41() #0 {
  ret i32 41
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f42() #0 {
  ret i32 42
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f43() #0 {
  ret i32 43
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f44() #0 {
  ret i32 44
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f45() #0 {
  ret i32 45
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f46() #0 {
  ret i32 46
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f47() #0 {
  ret i32 47
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f48() #0 {
  ret i32 48
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f49() #0 {
  ret i32 49
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f50() #0 {
  ret i32 50
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f51() #0 {
  ret i32 51
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f52() #0 {
  ret i32 52
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f53() #0 {
  ret i32 53
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f54() #0 {
  ret i32 54
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f55() #0 {
  ret i32 55
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f56() #0 {
  ret i32 56
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f57() #0 {
  ret i32 57
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f58() #0 {
  ret i32 58
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f59() #0 {
  ret i32 59
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f60() #0 {
  ret i32 60
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f61() #0 {
  ret i32 61
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f62() #0 {
  ret i32 62
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f63() #0 {
  ret i32 63
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f64() #0 {
  ret i32 64
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f65() #0 {
  ret i32 65
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f66() #0 {
  ret i32 66
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f67() #0 {
  ret i32 67
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f68() #0 {
  ret i32 68
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f69() #0 {
  ret i32 69
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f70() #0 {
  ret i32 70
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f71() #0 {
  ret i32 71
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f72() #0 {
  ret i32 72
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f73() #0 {
  ret i32 73
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f74() #0 {
  ret i32 74
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f75() #0 {
  ret i32 75
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f76() #0 {
  ret i32 76
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f77() #0 {
  ret i32 77
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f78() #0 {
  ret i32 78
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f79() #0 {
  ret i32 79
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f80() #0 {
  ret i32 80
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f81() #0 {
  ret i32 81
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f82() #0 {
  ret i32 82
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f83() #0 {
  ret i32 83
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f84() #0 {
  ret i32 84
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f85() #0 {
  ret i32 85
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f86() #0 {
  ret i32 86
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f87() #0 {
  ret i32 87
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f88() #0 {
  ret i32 88
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f89() #0 {
  ret i32 89
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f90() #0 {
  ret i32 90
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f91() #0 {
  ret i32 91
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f92() #0 {
  ret i32 92
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f93() #0 {
  ret i32 93
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f94() #0 {
  ret i32 94
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f95() #0 {
  ret i32 95
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f96() #0 {
  ret i32 96
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f97() #0 {
  ret i32 97
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f98() #0 {
  ret i32 98
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f99() #0 {
  ret i32 99
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f100() #0 {
  ret i32 100
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f101() #0 {
  ret i32 101
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f102() #0 {
  ret i32 102
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f103() #0 {
  ret i32 103
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f104() #0 {
  ret i32 104
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f105() #0 {
  ret i32 105
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f106() #0 {
  ret i32 106
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f107() #0 {
  ret i32 107
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f108() #0 {
  ret i32 108
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f109() #0 {
  ret i32 109
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f110() #0 {
  ret i32 110
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f111() #0 {
  ret i32 111
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f112() #0 {
  ret i32 112
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f113() #0 {
  ret i32 113
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f114() #0 {
  ret i32 114
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f115() #0 {
  ret i32 115
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f116() #0 {
  ret i32 116
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f117() #0 {
  ret i32 117
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f118() #0 {
  ret i32 118
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f119() #0 {
  ret i32 119
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f120() #0 {
  ret i32 120
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f121() #0 {
  ret i32 121
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f122() #0 {
  ret i32 122
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f123() #0 {
  ret i32 123
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f124() #0 {
  ret i32 124
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f125() #0 {
  ret i32 125
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f126() #0 {
  ret i32 126
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f127() #0 {
  ret i32 127
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f128() #0 {
  ret i32 128
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f129() #0 {
  ret i32 129
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f130() #0 {
  ret i32 130
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f131() #0 {
  ret i32 131
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f132() #0 {
  ret i32 132
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f133() #0 {
  ret i32 133
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f134() #0 {
  ret i32 134
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f135() #0 {
  ret i32 135
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f136() #0 {
  ret i32 136
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f137() #0 {
  ret i32 137
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f138() #0 {
  ret i32 138
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f139() #0 {
  ret i32 139
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f140() #0 {
  ret i32 140
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f141() #0 {
  ret i32 141
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f142() #0 {
  ret i32 142
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f143() #0 {
  ret i32 143
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f144() #0 {
  ret i32 144
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f145() #0 {
  ret i32 145
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f146() #0 {
  ret i32 146
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f147() #0 {
  ret i32 147
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f148() #0 {
  ret i32 148
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f149() #0 {
  ret i32 149
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f150() #0 {
  ret i32 150
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f151() #0 {
  ret i32 151
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f152() #0 {
  ret i32 152
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f153() #0 {
  ret i32 153
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f154() #0 {
  ret i32 154
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f155() #0 {
  ret i32 155
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f156() #0 {
  ret i32 156
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f157() #0 {
  ret i32 157
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f158() #0 {
  ret i32 158
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f159() #0 {
  ret i32 159
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f160() #0 {
  ret i32 160
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f161() #0 {
  ret i32 161
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f162() #0 {
  ret i32 162
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f163() #0 {
  ret i32 163
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f164() #0 {
  ret i32 164
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f165() #0 {
  ret i32 165
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f166() #0 {
  ret i32 166
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f167() #0 {
  ret i32 167
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f168() #0 {
  ret i32 168
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f169() #0 {
  ret i32 169
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f170() #0 {
  ret i32 170
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f171() #0 {
  ret i32 171
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f172() #0 {
  ret i32 172
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f173() #0 {
  ret i32 173
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f174() #0 {
  ret i32 174
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f175() #0 {
  ret i32 175
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f176() #0 {
  ret i32 176
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f177() #0 {
  ret i32 177
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f178() #0 {
  ret i32 178
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f179() #0 {
  ret i32 179
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f180() #0 {
  ret i32 180
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f181() #0 {
  ret i32 181
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f182() #0 {
  ret i32 182
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f183() #0 {
  ret i32 183
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f184() #0 {
  ret i32 184
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f185() #0 {
  ret i32 185
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f186() #0 {
  ret i32 186
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f187() #0 {
  ret i32 187
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f188() #0 {
  ret i32 188
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f189() #0 {
  ret i32 189
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f190() #0 {
  ret i32 190
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f191() #0 {
  ret i32 191
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f192() #0 {
  ret i32 192
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f193() #0 {
  ret i32 193
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f194() #0 {
  ret i32 194
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f195() #0 {
  ret i32 195
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f196() #0 {
  ret i32 196
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f197() #0 {
  ret i32 197
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f198() #0 {
  ret i32 198
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f199() #0 {
  ret i32 199
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f200() #0 {
  ret i32 200
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @print_str(ptr noundef @.str)
  %2 = call i32 @f123()
  %3 = sext i32 %2 to i64
  call void @print_num(i64 noundef %3)
  call void @newline()
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Debian clang version 19.1.7 (3+b1)"}
!6 = !{i64 100, i64 128}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
