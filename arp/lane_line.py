
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
    def isBlockLine(line):
        return line in [lane_line.BOUNDARY, lane_line.FORK_EDGE, lane_line.YSOLID, lane_line.YYSOILD, lane_line.YDASH, lane_line.YYDASH]

import lane_line

lane_line.BOUNDARY = "boundary"
lane_line.YSOLID = "yellow solid"
lane_line.YYSOILD = "yellow solid soild"
lane_line.YDASH = "yellow dash"
lane_line.YYDASH = "yellow dash dash"
lane_line.FORK_LINE = "fork_line"
lane_line.FORK_EDGE = "fork_edge"