class Params(dict):
    def __init__(self, *arg, **kw):
        super(Params, self).__init__(*arg, **kw)
        self.__dict__ = self


class ParamsIterator:

    def __init__(self, *args):
        self.param_lists = [a for a in args]
        if not self.param_lists:
            raise RuntimeError("Parameters cannot be empty")
        self.param_list_lens = [len(x)-1 for x in self.param_lists]
        self.curr_iters = [0 for _ in range(len(self.param_lists))]
        self.curr_iter_idx = 0

    def _update_iter(self):
        if self.iteration_done():
            return
        if self.curr_iters[self.curr_iter_idx] == self.param_list_lens[self.curr_iter_idx]:
            while self.curr_iter_idx < len(self.param_lists) \
                and self.curr_iters[self.curr_iter_idx] == self.param_list_lens[self.curr_iter_idx]:
                self.curr_iter_idx += 1
            if self.curr_iter_idx >= len(self.param_lists):
                assert(self.iteration_done())
            else:
                self.curr_iters[self.curr_iter_idx] += 1
                for j in range(self.curr_iter_idx):
                    self.curr_iters[j] = 0
                self.curr_iter_idx = 0
        else:
            self.curr_iters[self.curr_iter_idx] += 1

    def iter_started(self):
        return self.curr_iter_idx == 0

    def get_next_combinations(self):
        rv = []
        for i in range(len(self.param_lists)):
            val = self.param_lists[i][self.curr_iters[i]]
            rv.append(val)
        self._update_iter()
        return rv

    def iteration_done(self):
        return self.curr_iter_idx >= len(self.param_lists)

    def reset(self):
        self.curr_iters = [0 for _ in range(len(self.param_lists))]
        self.curr_iter_idx = 0


if __name__ == "__main__":

    it = ParamsIterator([1,2,3], ['a', 'b'], [-1, -2, -3, -4])

    while not it.iteration_done():
        li = it.get_next_combinations()
        print(li)

    print("")

    it = ParamsIterator([1, 2], ['a'])

    while not it.iteration_done():
        li = it.get_next_combinations()
        print(li)

