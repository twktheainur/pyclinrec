import ctypes
import os


class WordDivider:
    def __init__(self, filename):
        self.word_divider_tail = 0
        self.word_divider_buf = []
        if not filename:
            # sourcery skip: raise-specific-error
            raise Exception("contact developer")
        if filename == "NonAlphanumeric":
            self.word_divider_tail = 65
            self.word_divider_buf.extend(
                c
                for c in range(1, 128)
                if c not in range(ord("0"), ord("9") + 1)
                and c not in range(ord("a"), ord("z") + 1)
                and c not in range(ord("A"), ord("Z") + 1)
            )
            if len(self.word_divider_buf) != self.word_divider_tail:
                raise Exception("contact developer")
        elif filename != "None":
            try:
                with open(filename, "r") as file:
                    line_num = 0
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        line = line.strip("\r\n")
                        line_num += 1
                        if not line or line[0] == ";":
                            continue
                        delim_loc = line.find(";")
                        if delim_loc != -1:
                            line = line[:delim_loc]
                        code = int(line, 16)
                        if code == -1:
                            raise Exception(
                                f"Worddivider file line '{line_num}' has invalid hex"
                            )
                        if len(self.word_divider_buf) == self.word_divider_tail:
                            self.word_divider_tail += 1024
                        self.word_divider_buf.append(code)
                        if self.word_divider_tail == ctypes.c_int.max:
                            raise Exception("Not a correct word-divider file")
                    if self.word_divider_tail == 0:
                        raise Exception("Empty word-divider file is not allowed")
                    self.word_divider_buf.sort()
            except FileNotFoundError as e:
                raise Exception(f"{filename}: No such a file") from e

    def __del__(self):
        self.word_divider_buf = None

    def is_boundary(self, s):
        original = s
        code = self.read_unicode_or_die(s)
        if self.is_divider(code):
            return True
        original = original[:-1]
        while ord(original[-1]) >> 6 == -2:
            original = original[:-1]
        code = self.read_unicode_or_die(original)
        return self.is_divider(code)

    def is_divider(self, code):
        return code == 0 or code in self.word_divider_buf

    def get_total(self):
        return len(self.word_divider_buf)

    def copy_divider(self, idx, buf):
        assert idx < len(self.word_divider_buf)
        utf8 = self.word_divider_buf[idx]
        # Conversion to UTF-8 not implemented here, requires additional context
        raise NotImplementedError("UTF-8 conversion not implemented")

    def read_unicode_or_die(self, s):
        # Unicode reading not implemented here, requires additional context
        raise NotImplementedError("Unicode reading not implemented")
