from datetime import datetime as dt

class Numerals:
    def __init__(self,max_num,min_num,exact=False,least=False,most=False):
        uncertain = .4
        self.max_num = max_num
        self.min_num = min_num

        if exact:
            self.max_num = self.max_num * (1 + .5 * uncertain)
            self.min_num = self.min_num * (1 - .5 * uncertain)

        if least:
            self.max_num = self.max_num * (1 + uncertain)
        if most:
            self.min_num = self.min_num * (1 - uncertain)

    def get_avg(self):
        return (self.max_num + self.min_num)/2
            
    def is_equals(self,n2):
        return self.max_num == n2.max_num and self.min_num == n2.min_num

    def add_nums(self,num2):
        return Numerals(self.max_num + num2.max_num,self.min_num + num2.min_num)

class Ordinal:
    def __init__(self,numeric):
        self.numeric = numeric

class YearVal:
    def __init__(self,numeric):
        self.numeric = numeric

class Unit:
    def __init__(self,min_hr,max_hr,avg_hr = None,plural=False):
        self.min_hr = min_hr
        self.max_hr = max_hr
        if avg_hr is None:
            self.avg_hr = (max_hr + min_hr)/2
        else:
            self.avg_hr = avg_hr
        self.plural = plural

    def is_equals(self,u2):
        return self.max_hr == u2.max_hr and self.min_hr == u2.min_hr and self.avg_hr == u2.avg_hr

class DateVal:
    def __init__(self,dist,unit):
        self.dist = dist
        self.length = unit

class TimeLength:
    def __init__(self,unit,num=Numerals(1,1)):
        self.num = num
        self.unit = unit

    def get_length(self):
        len_max = self.num.max_num * self.unit.avg_hr
        len_min = self.num.min_num * self.unit.avg_hr
        return Numerals(len_max,len_min)

    def is_equals(self,tl2):
        return self.num.is_equals(tl2.num) and self.unit.is_equals(tl2.unit)

    def check_type(self):
        if not self.num.is_equals(Numerals(1,1)):
            return None

        if self.unit.is_equals(Unit(0,0)):
            return "Minute"
        if self.unit.is_equals(Unit(1,1)):
            return "Hour"
        if self.unit.is_equals(Unit(24,24)):
            return "Day"
        if self.unit.is_equals(Unit(168,168)):
            return "Week"
        if self.unit.is_equals(Unit(672,744,avg_hr=731)):
            return "Month"
        if self.unit.is_equals(Unit(8760,8784,avg_hr=8766)):
            return "Year"

        return None
        
class Frequency:
    def __init__(self,denominator,numerator=Numerals(1,1)):
        self.numerator = numerator
        self.denominator = denominator

class CycleType:
    def __init__(self,period,step_len):
        self.period = period
        self.step_len = step_len
        
class Cycle:
    def __init__(self,cycle_type,val,extra=None):
        self.cycle_type = cycle_type
        self.val = val

class IntRange:
    def __init__(self,int_or_rng,anchor_s=None,anchor_e=None,length=None,interval=None):
        self.int_or_rng = int_or_rng
        self.anchor_s = anchor_s
        self.anchor_e = anchor_e        
        self.length = length
        # Interval here is just a TimeLength
        if int_or_rng == "Range":
            self.interval = interval
        else:
            self.interval = None

    def get_length(self):
        return self.length.get_length()

class Point:
    def __init__(self,anchor=None,direct="Present",dist=0,length=None):
        self.anchor = anchor
        self.direct = direct
        self.dist = dist
        self.length = length

    def get_dist(self):
        if isinstance(self.dist,TimeLength):
            self_dist = self.dist.get_length()
        elif isinstance(self.dist,Numerals):
            self_dist = self.dist
        else:
            self_dist = Numerals(self.dist,self.dist)

        if self.direct == "Past":
            return Numerals(-1 * self_dist.max_num,-1 * self_dist.min_num)
        return self_dist

    def reduce_self(self):
        if self.anchor == "Present" or self.anchor == "Zero Date":
            return self

        self_dist = self.get_dist()
        anchor_dist = self.anchor.get_dist()
        new_dist = self.dist.add_nums(anchor_dist)

        if new_dist.max_num < 0:
            final_dist = Numerals(-1 * new_dist.max_num,-1 * new_dist.min_num)
            return Point(anchor=self.anchor.anchor,direct="Past",dist=final_dist)
        else:
            return Point(anchor=self.anchor.anchor,direct="Future",dist=new_dist
)

class Recurrence:
    def __init__(self,frequency=None,anchors=None,rangex=None,length=None):
        self.frequency = frequency
        self.anchors = anchors
        self.rangex = rangex
        self.length = length

class DateCapture:
    def __init__(self,datetime):
        self.year = datetime.year
        self.month = datetime.month
        self.day = datetime.day
        self.weekday = datetime.weekday()

class Partial:
    def __init__(self,rulename,timeexp):
        self.rulename = rulename
        self.timeexp = timeexp
