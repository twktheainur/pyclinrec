class TreeNode:
    def __init__(self, word, id_ptr, id_size):
        if not word:
            raise ValueError("Empty/Null word is not allowed")
        self.word_ = word
        self.right_ = None
        self.down_ = None
        self.init_id_buf(id_ptr, id_size)

    def __del__(self):
        # Python's garbage collector handles memory management,
        # so explicit deletion is not necessary here.
        pass

    def insert_word(self, word, id_ptr, id_size):
        pRtn = self
        idx = 0
        for me, you in zip(self.word_, word):
            if me != you or not me:
                break
            idx += 1
        else:
            me = self.word_[idx] if self.word_[idx:] else "\0"
            you = "\0"

        if idx == 0:
            if me < you:
                if self.down_:
                    self.down_ = self.down_.insert_word(word, id_ptr, id_size)
                else:
                    self.down_ = self.new_instance(word, id_ptr, id_size)
            else:
                pRtn = self.new_instance(word, id_ptr, id_size)
                pRtn.down_ = self
        elif me != "\0":
            pTmp = self.new_instance(self.word_[idx:], None, 0)
            pTmp.id_buf_ = self.id_buf_
            pTmp.right_ = self.right_
            self.word_ = self.word_[:idx]
            self.right_ = pTmp
            if you == "\0":
                self.init_id_buf(id_ptr, id_size)
            else:
                self.id_buf_ = None
                self.right_ = self.right_.insert_word(word[idx:], id_ptr, id_size)
        elif you == "\0":
            original_id_size = self.get_id_size()
            self.id_buf_ = bytearray(self.id_buf_)
            self.id_buf_.extend(bytearray(id_size + original_id_size))
            self.copy_id_buf(original_id_size, id_ptr, id_size)
        elif self.right_:
            self.right_ = self.right_.insert_word(word[idx:], id_ptr, id_size)
        else:
            self.right_ = self.new_instance(word[idx:], id_ptr, id_size)
        return pRtn

    def word(self):
        return self.word_

    def down(self):
        return self.down_

    def right(self):
        return self.right_

    def get_id_size(self):
        return len(self.id_buf_) if self.id_buf_ else 0

    def get_id_element(self, idx):
        return self.id_buf_[idx]

    def new_instance(self, word, id_ptr, id_size):
        return TreeNode(word, id_ptr, id_size)

    def copy_id_buf(self, to_location, copy_from, id_size):
        self.id_buf_[to_location : to_location + id_size] = copy_from[:id_size]

    def init_id_buf(self, id_ptr, id_size):
        if id_size:
            self.id_buf_ = bytearray(id_size)
            self.copy_id_buf(0, id_ptr, id_size)
        else:
            self.id_buf_ = None
