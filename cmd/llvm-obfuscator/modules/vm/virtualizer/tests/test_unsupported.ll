; ModuleID = 'test_unsupported'
source_filename = "test_unsupported.c"

; Function with call - should be SKIPPED
declare i32 @other_func(i32)

define i32 @with_call(i32 %a) {
entry:
  %result = call i32 @other_func(i32 %a)
  ret i32 %result
}

; Function with branch - should be SKIPPED
define i32 @with_branch(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %positive, label %negative

positive:
  ret i32 1

negative:
  ret i32 0
}

; Function with load/store - should be SKIPPED
define i32 @with_memory(i32* %ptr) {
entry:
  %val = load i32, i32* %ptr
  ret i32 %val
}

; Simple function that SHOULD work (control test)
define i32 @simple_add(i32 %x, i32 %y) {
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}
