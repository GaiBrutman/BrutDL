class PrintUtils:
    @staticmethod
    def line_print(*args):
        print('\r', end='', flush=True)
        print(''.join(args), end='', flush=True)

    @staticmethod
    def progress_str(i, iter_max=1, print_max=30):
        progress_num = int(print_max * (i + 1) / iter_max)
        progress_str = '[' + '=' * progress_num + '>' + ' ' * (print_max - progress_num) + ']'
        return progress_str

    @staticmethod
    def print_progress(i, iter_max, epoch, cost, *args):
        """
        Prints the progress of the training in the same line.
        :param i: current iteration
        :param iter_max: iteration limit
        :param epoch: current epoch
        :param cost: cost value
        :param args: additional arguments (Strings).
        :return: None
        """

        progress_visual = PrintUtils.progress_str(i, iter_max=iter_max)
        s = 'epoch %i ' % epoch + progress_visual + ' Cost: %f' % float(cost)

        PrintUtils.line_print(s, *[', ' + a for a in args if a])

    @staticmethod
    def hyphen_line(n):
        return '-' * n + '\n'
