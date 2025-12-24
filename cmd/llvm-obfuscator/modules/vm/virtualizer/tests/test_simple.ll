; ModuleID = 'test_simple'
source_filename = "test_simple.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Simple add function - should be virtualized
define i32 @add_numbers(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

; Simple sub function - should be virtualized
define i32 @sub_numbers(i32 %a, i32 %b) {
entry:
  %diff = sub i32 %a, %b
  ret i32 %diff
}

; Simple xor function - should be virtualized
define i32 @xor_numbers(i32 %a, i32 %b) {
entry:
  %result = xor i32 %a, %b
  ret i32 %result
}

; Combined operations - should be virtualized
define i32 @combined_ops(i32 %a, i32 %b, i32 %c) {
entry:
  %sum = add i32 %a, %b
  %result = sub i32 %sum, %c
  ret i32 %result
}
