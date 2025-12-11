//precision 0
//epsilon 1e-99

//dupe false
//simplify true

//global test_4 5

# test 1: normal boring repeat
//repeat 5
//store A
//ret + A 1
//endrepeat

# test 2: repeat with locals
//repeat 5
//local inc 1
//store B
//ret + B inc
//endrepeat

# test 3: indexed repeats with locals
//repeat 5 i
//local inc i
//store C
//ret + C inc
//endrepeat

# test 4: pythonic and paramath indexed repeat with locals
//repeat test_4 i
//local pythonic len("hello world") + loop.i
//local paramath (max i 2)
//local pythonreference local.paramath ** 3
//store D
//ret + D pythonic paramath pythonreference
//endrepeat
