//precision 5
//epsilon 0

# test 1: global scope basic eval
//global length len("silly")

# test 2: referring globals in eval
//global area globals.length ** 2

# local scope
//repeat 5 i
# test 3: local scope basic eval
//local one_third 1/3

# test 4: referring python globals in eval
//local cube globals.area * i

# test 5 referring python locals in eval
//local pyramid globals.area * i * locals.one_third

# nice way to end things off
//store A
//ret pyramid

//endrepeat