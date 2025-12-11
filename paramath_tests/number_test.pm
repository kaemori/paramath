//alias TESTNUM A 
//alias CHECKSUM B 

//precision 0
//epsilon 1e-99
//simplify true
//dupe false

//global const_A 123456
//global len_A len(str(consts.const_A))

//store TESTNUM
//ret const_A

//store CHECKSUM
//ret 0

//repeat len_A i
//local div_len 10 ** (consts.len_A - loop.i - 1)
//local test_var str(consts.const_A)[loop.i+1: ] or 0

//store CHECKSUM
target = nat(
    /
    TESTNUM div_len
)
difference = - target test_var
//ret + CHECKSUM difference
//endrepeat

//display
//ret - len_A CHECKSUM
