""" Testing some IAI functionalities. """

function test_kwargs()
    lnr = base_otc()
    grid = gridify(lnr)
    
    kwargs = default_fit_kwargs()
end