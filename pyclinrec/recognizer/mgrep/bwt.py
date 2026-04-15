import threading
from typing import List, Tuple, Optional


class BWT:
    kMinRangePerThread = 1024
    kMinNmembPerThread = 2000000

    def __init__(self):
        self.first_ = None
        self.text_length_ = None
        self.bucket_counter_ = None
        self.last_ = None
        self.bit_of_bucket_size_ = None

    def search(
        self,
        word_id: int,
        line: str,
        word: str,
        word_divider,
        max_parallel_depth: int,
        mismatch_limit: int,
    ):
        print(f"#{line}")
        word_len = len(word)
        if not word_len:
            return
        word_buf = bytearray(word, "utf-8") + bytearray(7)
        matched_size = word_len + mismatch_limit + 12  # Adjusted for 2*UTF8 chars
        matched_buf = bytearray(matched_size)
        result_set = ResultSet()
        for i in range(-2, word_divider.get_total()):
            if i == -2:
                word_buf[word_len] = ord("\n")
                divider_len = 1
            elif i == -1:
                word_buf[word_len] = 0
                divider_len = 1
            else:
                divider_len = word_divider.copy_divider(i, word_buf[word_len:])
            word_buf[word_len + divider_len] = 0
            assert divider_len <= 6
            c = word_buf[word_len + divider_len - 1]
            matched_buf[0] = c
            self._fuzzy_search(
                max_parallel_depth,
                0,
                "",
                word_divider,
                word_buf.decode("utf-8"),
                word_len,
                word_len + divider_len - 1,
                self.first_[c - 1] if c else 0,
                self.first_[c],
                matched_buf.decode("utf-8"),
                matched_size,
                1,
                mismatch_limit,
                0,
                0,
                divider_len,
                result_set,
            )
        result_set.print(word_id)

    def read_index(self, findex):
        pass  # Implementation of read_index method

    def build_index(
        self,
        text_buf: str,
        path: str,
        findex,
        memory_size: int,
        max_parallel_depth: int,
        bit_of_bucket_size: str,
    ):
        pass  # Implementation of build_index method

    def _fuzzy_search_thread_or_not(
        self,
        max_parallel_depth: int,
        cur_parallel_depth: int,
        skipped: str,
        worddivider,
        word_buf: str,
        divider_pos: int,
        word_tail: int,
        last_from: int,
        last_to: int,
        matched_buf: str,
        matched_size: int,
        matched_tail: int,
        mismatch_limit: int,
        mismatch_num: int,
        leading_divider_len: int,
        trailing_divider_len: int,
        result_set,
        th_arr: Optional[List[threading.Thread]],
        th_size: List[int],
        th_tail: List[int],
    ):
        if (
            cur_parallel_depth <= max_parallel_depth
            and last_from + self.kMinRangePerThread < last_to
        ):
            arg = (
                self,
                max_parallel_depth,
                cur_parallel_depth,
                skipped,
                worddivider,
                word_buf,
                divider_pos,
                word_tail,
                last_from,
                last_to,
                matched_buf,
                matched_size,
                matched_tail,
                mismatch_limit,
                mismatch_num,
                leading_divider_len,
                trailing_divider_len,
                result_set,
            )
            if len(th_arr) == th_tail:
                th_arr.extend([None] * 64)  # Increase thread array size
            try:
                th_arr[th_tail] = threading.Thread(
                    target=self._fuzzy_search_thread_worker, args=(arg,)
                )
                th_arr[th_tail].start()
                th_tail += 1
            except Exception as e:
                th_tail -= 1
                print(f"Error creating thread: {e}, try smaller 'parallel-depth'.")
        else:
            self._fuzzy_search(
                max_parallel_depth,
                cur_parallel_depth,
                skipped,
                worddivider,
                word_buf,
                divider_pos,
                word_tail,
                last_from,
                last_to,
                matched_buf,
                matched_size,
                matched_tail,
                mismatch_limit,
                mismatch_num,
                leading_divider_len,
                trailing_divider_len,
                result_set,
            )

    @staticmethod
    def _fuzzy_search_thread_worker(arg):
        pass  # Implementation of fuzzy_search_thread_worker method

    def _write_char(
        self,
        findex,
        c: int,
        written_bytes: int,
        bit_of_bucket_size: str,
        counter_per_char: List[int],
    ):
        pass  # Implementation of write_char method

    def _read_char(self, text_length: int, text_buf: str, idx: int, shift: int) -> int:
        pass  # Implementation of read_char method

    def _quick_sort(
        self,
        text_length: int,
        text_buf: str,
        max_parallel_depth: int,
        base: List[int],
        nmemb: int,
        shift: int,
        depth: int,
    ):
        pass  # Implementation of quick_sort method

    def _qs_thread_or_not(
        self,
        text_length: int,
        text_buf: str,
        max_parallel_depth: int,
        base: List[int],
        nmemb: int,
        shift: int,
        depth: int,
        th_arr: Optional[List[threading.Thread]],
        th_size: List[int],
        th_tail: List[int],
    ):
        pass  # Implementation of qs_thread_or_not method

    @staticmethod
    def _qs_thread_worker(arg):
        pass  # Implementation of qs_thread_worker method


class ResultSet:
    def __init__(self):
        self.result_arr_ = []
        self.result_tail_ = 0
        self.result_size_ = 0
        self.mutex_ = threading.Lock()

    def append(
        self,
        matched_buf: str,
        matched_tail: int,
        count: int,
        mismatch: int,
        leading_divider_len: int,
        trailing_divider_len: int,
    ):
        pass  # Implementation of append method

    def print(self, word_id: int):
        pass  # Implementation of print method


class Result:
    def __init__(
        self,
        matched_buf: str,
        matched_tail: int,
        count: int,
        mismatch: int,
        leading_divider_len: int,
        trailing_divider_len: int,
    ):
        self.matched_buf_ = matched_buf
        self.matched_tail_ = matched_tail
        self.count_ = count
        self.mismatch_ = mismatch
        self.leading_divider_len_ = leading_divider_len
        self.trailing_divider_len_ = trailing_divider_len

    def print(self, word_id: int):
        pass  # Implementation of print method

    @staticmethod
    def diff(ano):
        pass  # Implementation of diff method

    @staticmethod
    def compare_buffer(p1, p2) -> int:
        pass  # Implementation of compare_buffer method

    @staticmethod
    def compare_count(p1, p2) -> int:
        pass  # Implementation of compare_count method
