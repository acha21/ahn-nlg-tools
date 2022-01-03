import unittest
from unittest import TestCase
import angt.eval.dstc as dstc


class DSTCTest(TestCase):
    def test_foo(self):
        self.assertEqual(dstc.foo(2,3), 5)


if __name__ == "__main__":
    unittest.main()
