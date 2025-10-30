import originpro as op


def interact_with_origin(func):
    """Decorator for correct work with originpro lib"""

    def wrapper(*args, **kwargs):
        # Ensures that the Origin instance gets shut down properly.
        print('Connecting to Origin, please wait 5 seconds...')
        import sys

        def origin_shutdown_exception_hook(exctype, value, traceback):
            op.exit()
            sys.__excepthook__(exctype, value, traceback)

        if op and op.oext:
            sys.excepthook = origin_shutdown_exception_hook

        # Set Origin instance visibility.
        if op.oext:
            op.set_show(True)

        func(*args, **kwargs)

        # Exit running instance of Origin.
        if op.oext:
            op.exit()

    return wrapper
