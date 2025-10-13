; ModuleID = 'mlir_test_output/layer4_ollvm.bc'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
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
@x = common global i32 0
@y = common global i32 0
@x.1 = common global i32 0
@y.2 = common global i32 0
@x.3 = common global i32 0
@y.4 = common global i32 0
@x.5 = common global i32 0
@y.6 = common global i32 0
@x.7 = common global i32 0
@y.8 = common global i32 0
@x.9 = common global i32 0
@y.10 = common global i32 0

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
  %2 = load i32, ptr @x.1, align 4
  %3 = load i32, ptr @y.2, align 4
  %4 = sub i32 %2, 1
  %5 = mul i32 %2, %4
  %6 = urem i32 %5, 2
  %7 = icmp eq i32 %6, 0
  %8 = icmp slt i32 %3, 10
  %9 = or i1 %7, %8
  br i1 %9, label %originalBB, label %originalBBalteredBB

originalBB:                                       ; preds = %1, %originalBBalteredBB
  %10 = alloca ptr, align 8
  store ptr %0, ptr %10, align 8
  %11 = load ptr, ptr %10, align 8
  %12 = load ptr, ptr @API_KEY, align 8
  %13 = call i32 @strcmp(ptr noundef %11, ptr noundef %12) #3
  %14 = icmp eq i32 %13, 0
  %15 = zext i1 %14 to i32
  %16 = load i32, ptr @x.1, align 4
  %17 = load i32, ptr @y.2, align 4
  %18 = sub i32 %16, 1
  %19 = mul i32 %16, %18
  %20 = urem i32 %19, 2
  %21 = icmp eq i32 %20, 0
  %22 = icmp slt i32 %17, 10
  %23 = or i1 %21, %22
  br i1 %23, label %originalBBpart2, label %originalBBalteredBB

originalBBpart2:                                  ; preds = %originalBB
  ret i32 %15

originalBBalteredBB:                              ; preds = %originalBB, %1
  %24 = alloca ptr, align 8
  store ptr %0, ptr %24, align 8
  %25 = load ptr, ptr %24, align 8
  %26 = load ptr, ptr @API_KEY, align 8
  %27 = call i32 @strcmp(ptr noundef %25, ptr noundef %26) #3
  %28 = icmp eq i32 %27, 0
  %29 = zext i1 %28 to i32
  br label %originalBB
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
  %2 = load i32, ptr @x.5, align 4
  %3 = load i32, ptr @y.6, align 4
  %4 = sub i32 %2, 1
  %5 = mul i32 %2, %4
  %6 = urem i32 %5, 2
  %7 = icmp eq i32 %6, 0
  %8 = icmp slt i32 %3, 10
  %9 = or i1 %7, %8
  br i1 %9, label %originalBB, label %originalBBalteredBB

originalBB:                                       ; preds = %1, %originalBBalteredBB
  %10 = alloca float, align 4
  store float %0, ptr %10, align 4
  %11 = load float, ptr %10, align 4
  %12 = fmul float %11, 0x400921FA00000000
  %13 = load i32, ptr @x.5, align 4
  %14 = load i32, ptr @y.6, align 4
  %15 = sub i32 %13, 1
  %16 = mul i32 %13, %15
  %17 = urem i32 %16, 2
  %18 = icmp eq i32 %17, 0
  %19 = icmp slt i32 %14, 10
  %20 = or i1 %18, %19
  br i1 %20, label %originalBBpart2, label %originalBBalteredBB

originalBBpart2:                                  ; preds = %originalBB
  ret float %12

originalBBalteredBB:                              ; preds = %originalBB, %1
  %21 = alloca float, align 4
  store float %0, ptr %21, align 4
  %22 = load float, ptr %21, align 4
  %_ = fneg float %22
  %gen = fadd float %_, 0x400921FA00000000
  %_1 = fsub float %22, 0x400921FA00000000
  %gen2 = fmul float %_1, 0x400921FA00000000
  %_3 = fsub float %22, 0x400921FA00000000
  %gen4 = fmul float %_3, 0x400921FA00000000
  %_5 = fsub float %22, 0x400921FA00000000
  %gen6 = fmul float %_5, 0x400921FA00000000
  %23 = fmul float %22, 0x400921FA00000000
  br label %originalBB
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f_b610cf36(ptr noundef %0, ptr noundef %1) #0 {
  %3 = load i32, ptr @x.7, align 4
  %4 = load i32, ptr @y.8, align 4
  %5 = sub i32 %3, 1
  %6 = mul i32 %3, %5
  %7 = urem i32 %6, 2
  %8 = icmp eq i32 %7, 0
  %9 = icmp slt i32 %4, 10
  %10 = or i1 %8, %9
  br i1 %10, label %originalBB, label %originalBBalteredBB

originalBB:                                       ; preds = %2, %originalBBalteredBB
  %.reg2mem = alloca i1, align 1
  %11 = alloca i32, align 4
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca i32, align 4
  %15 = alloca float, align 4
  store ptr %0, ptr %12, align 8
  store ptr %1, ptr %13, align 8
  %16 = load ptr, ptr %12, align 8
  %17 = call i32 @f_61ae7c24(ptr noundef %16)
  %18 = icmp ne i32 %17, 0
  store i1 %18, ptr %.reg2mem, align 1
  %switchVar = alloca i32, align 4
  store i32 1163962479, ptr %switchVar, align 4
  %19 = load i32, ptr @x.7, align 4
  %20 = load i32, ptr @y.8, align 4
  %21 = sub i32 %19, 1
  %22 = mul i32 %19, %21
  %23 = urem i32 %22, 2
  %24 = icmp eq i32 %23, 0
  %25 = icmp slt i32 %20, 10
  %26 = or i1 %24, %25
  br i1 %26, label %originalBBpart2, label %originalBBalteredBB

originalBBpart2:                                  ; preds = %originalBB
  br label %loopEntry

loopEntry:                                        ; preds = %originalBBpart2, %loopEnd
  %switchVar1 = load i32, ptr %switchVar, align 4
  switch i32 %switchVar1, label %switchDefault [
    i32 1163962479, label %first
    i32 1793210059, label %44
    i32 2024780842, label %49
    i32 -1465529257, label %58
    i32 780605168, label %60
    i32 1129575378, label %62
  ]

switchDefault:                                    ; preds = %loopEntry
  br label %loopEnd

first:                                            ; preds = %loopEntry
  %27 = load i32, ptr @x.7, align 4
  %28 = load i32, ptr @y.8, align 4
  %29 = sub i32 %27, 1
  %30 = mul i32 %27, %29
  %31 = urem i32 %30, 2
  %32 = icmp eq i32 %31, 0
  %33 = icmp slt i32 %28, 10
  %34 = or i1 %32, %33
  br i1 %34, label %originalBB2, label %originalBB2alteredBB

originalBB2:                                      ; preds = %first, %originalBB2alteredBB
  %.reload = load i1, ptr %.reg2mem, align 1
  %35 = select i1 %.reload, i32 1793210059, i32 780605168
  store i32 %35, ptr %switchVar, align 4
  %36 = load i32, ptr @x.7, align 4
  %37 = load i32, ptr @y.8, align 4
  %38 = sub i32 %36, 1
  %39 = mul i32 %36, %38
  %40 = urem i32 %39, 2
  %41 = icmp eq i32 %40, 0
  %42 = icmp slt i32 %37, 10
  %43 = or i1 %41, %42
  br i1 %43, label %originalBBpart24, label %originalBB2alteredBB

originalBBpart24:                                 ; preds = %originalBB2
  br label %loopEnd

44:                                               ; preds = %loopEntry
  %45 = load ptr, ptr %13, align 8
  %46 = call i32 @f_0218a827(ptr noundef %45)
  %47 = icmp ne i32 %46, 0
  %48 = select i1 %47, i32 2024780842, i32 -1465529257
  store i32 %48, ptr %switchVar, align 4
  br label %loopEnd

49:                                               ; preds = %loopEntry
  %50 = call i32 @f_f8b7e158(i32 noundef 42)
  store i32 %50, ptr %14, align 4
  %51 = load i32, ptr %14, align 4
  %52 = sitofp i32 %51 to float
  %53 = call float @f_b9127cd9(float noundef %52)
  store float %53, ptr %15, align 4
  %54 = load i32, ptr %14, align 4
  %55 = load float, ptr %15, align 4
  %56 = fpext float %55 to double
  %57 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i32 noundef %54, double noundef %56)
  store i32 1, ptr %11, align 4
  store i32 1129575378, ptr %switchVar, align 4
  br label %loopEnd

58:                                               ; preds = %loopEntry
  %59 = call i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i32 0, ptr %11, align 4
  store i32 1129575378, ptr %switchVar, align 4
  br label %loopEnd

60:                                               ; preds = %loopEntry
  %61 = call i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i32 0, ptr %11, align 4
  store i32 1129575378, ptr %switchVar, align 4
  br label %loopEnd

62:                                               ; preds = %loopEntry
  %63 = load i32, ptr %11, align 4
  ret i32 %63

loopEnd:                                          ; preds = %60, %58, %49, %44, %originalBBpart24, %switchDefault
  br label %loopEntry

originalBBalteredBB:                              ; preds = %originalBB, %2
  %.reg2memalteredBB = alloca i1, align 1
  %64 = alloca i32, align 4
  %65 = alloca ptr, align 8
  %66 = alloca ptr, align 8
  %67 = alloca i32, align 4
  %68 = alloca float, align 4
  store ptr %0, ptr %65, align 8
  store ptr %1, ptr %66, align 8
  %69 = load ptr, ptr %65, align 8
  %70 = call i32 @f_61ae7c24(ptr noundef %69)
  %71 = icmp ne i32 %70, 0
  store i1 %71, ptr %.reg2memalteredBB, align 1
  %switchVaralteredBB = alloca i32, align 4
  store i32 1163962479, ptr %switchVaralteredBB, align 4
  br label %originalBB

originalBB2alteredBB:                             ; preds = %originalBB2, %first
  %.reloadalteredBB = load i1, ptr %.reg2mem, align 1
  %72 = select i1 %.reloadalteredBB, i32 1793210059, i32 780605168
  store i32 %72, ptr %switchVar, align 4
  br label %originalBB2
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
  %6 = call i32 @f_b610cf36(ptr noundef @.str, ptr noundef @.str.1)
  store i32 %6, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  %8 = call i32 (ptr, ...) @printf(ptr noundef @.str.8, i32 noundef %7)
  %9 = call i32 @f_b610cf36(ptr noundef @.str.9, ptr noundef @.str.10)
  store i32 %9, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  %11 = call i32 (ptr, ...) @printf(ptr noundef @.str.11, i32 noundef %10)
  %12 = call i32 (ptr, ...) @printf(ptr noundef @.str.12)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #3 = { nounwind willreturn memory(read) }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"clang version 22.0.0git (https://github.com/SkySingh04/llvm-project.git eba35723958cf6da53e6ac7c8223ce914047bca4)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
