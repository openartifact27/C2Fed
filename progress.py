"""Progress iterator shared by experiment entry points."""


def pit(it, *pargs, **nargs):
    import enlighten

    global __pit_man__
    try:
        __pit_man__
    except NameError:
        __pit_man__ = enlighten.get_manager()
    man = __pit_man__
    try:
        it_len = len(it)
    except Exception:
        it_len = None
    try:
        ctr = None
        for i, e in enumerate(it):
            if i == 0:
                ctr = man.counter(*pargs, **{**dict(leave=False, total=it_len), **nargs})
            yield e
            ctr.update()
    finally:
        if ctr is not None:
            ctr.close()
