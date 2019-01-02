
import time


class Line:
    class ConstError(TypeError) : pass
    class ConstCaseError(ConstError):pass

    def __setattr__(self, name, value):
            if name in self.__dict__:
                raise self.ConstError, "Can't change const value!"
            if not name.isupper():
                raise self.ConstCaseError, 'const "%s" is not all letters are capitalized' %name
            self.__dict__[name] = value

    @staticmethod
    def isBlockLine(self, line):
        return line in [Line.BOUNDARY, Line.FORK_EDGE, Line.YSOLID, Line.YYSOILD, Line.YDASH, Line.YYDASH]

Line.BOUNDARY = "boundary"
Line.YSOLID = "yellow solid"
Line.YYSOILD = "yellow solid soild"
Line.YDASH = "yellow dash"
Line.YYDASH = "yellow dash dash"
Line.FORK_LINE = "fork_line"
Line.FORK_EDGE = "fork_edge"