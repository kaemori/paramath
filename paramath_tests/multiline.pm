//precision 0
//epsilon 1e-99

//simplify true
//dupe false

# test 0: normal line
//display
//ret + A 6

# test 1: single equation multiline
//display
//ret (+
    A
    6
)

# test 2: multiline nested
//display
//ret (+
    A
    (- 6 B)
)

# test 3: all multiline
//display
//ret (+
    A
    (-
        6
        B
    )
)