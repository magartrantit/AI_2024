import enum

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from reteaua import Network, CrossEntropyCost, RELu

class DefaultEnum(enum.IntEnum):
  def to_neural_network_input(self):
    smallest, greatest = None, None
    for e in type(self).__members__.values():
      e = int(e)
      if smallest is None or smallest > e: smallest = e
      if greatest is None or greatest < e: greatest = e
    return [(int(self) - smallest) / (greatest - smallest)]

class Gender(DefaultEnum):
  MALE = enum.auto()
  FEMALE = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'Gender':
    if raw is None:
      raise MissingValueException("gender")
    normalized = raw.lower()
    match normalized:
      case 'male' | 'm':    return Gender.MALE
      case 'female' | 'f':  return Gender.FEMALE
      case 'nsp':           raise MissingValueException("gender")
      case _:               raise InvalidInputException('gender', normalized, 'string representation')

class Age(DefaultEnum):
  LESS_THAN_1_YEAR = enum.auto()                            # [0, 1)
  AT_LEAST_1_YEAR_BUT_LESS_THAN_2_YEARS = enum.auto()       # [1, 2)
  AT_LEAST_2_YEAR_BUT_LESS_THAN_10_YEARS = enum.auto()      # [2, 10)
  AT_LEAST_10_YEARS = enum.auto()                           # [10, inf)

  @staticmethod
  def from_string(raw: str) -> 'Age':
    if raw is None:
      raise MissingValueException("age")
    normalized = raw.lower()
    match normalized:
      case 'moinsde1':  return Age.LESS_THAN_1_YEAR
      case '1a2':       return Age.AT_LEAST_1_YEAR_BUT_LESS_THAN_2_YEARS
      case '2a10':      return Age.AT_LEAST_2_YEAR_BUT_LESS_THAN_10_YEARS
      case 'plusde10':  return Age.AT_LEAST_10_YEARS
      case _:           raise InvalidInputException('age', normalized, 'string representation')

  @staticmethod
  def from_float(age: float) -> 'Age':
    if          age < 0.0:  raise InvalidInputException('gender', age)
    elif 0.0 <= age < 1.0:  return Age.LESS_THAN_1_YEAR
    elif 1.0 <= age < 2.0:  return Age.AT_LEAST_1_YEAR_BUT_LESS_THAN_2_YEARS
    elif 2.0 <= age < 10.0: return Age.AT_LEAST_2_YEAR_BUT_LESS_THAN_10_YEARS
    else:                   return Age.AT_LEAST_10_YEARS

  @staticmethod
  def from_int(age: int) -> 'Age':
    return Age.from_float(float(age))

class Race(enum.IntEnum):
  BENGAL = enum.auto()
  BIRMAN = enum.auto()
  BRITISH_SHORTHAIR = enum.auto()
  CHARTREUX = enum.auto()
  EUROPEAN = enum.auto()
  MAINE_COON = enum.auto()
  PERSIAN = enum.auto()
  RAGDOLL = enum.auto()
  SAVANNAH = enum.auto()
  SPHYNX = enum.auto()
  SIAMESE = enum.auto()
  TURKISH_ANGORA = enum.auto()
  OTHER = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'Race':
    if raw is None:
      raise MissingValueException("race", False)
    normalized = raw.lower()
    match normalized:
      case 'ben':           return Race.BENGAL
      case 'sbi':           return Race.BIRMAN
      case 'bri':           return Race.BRITISH_SHORTHAIR
      case 'cha':           return Race.CHARTREUX
      case 'eur':           return Race.EUROPEAN
      case 'mco':           return Race.MAINE_COON
      case 'per':           return Race.PERSIAN
      case 'rag':           return Race.RAGDOLL
      case 'sav':           return Race.SAVANNAH
      case 'sph':           return Race.SPHYNX
      case 'ori':           return Race.SIAMESE
      case 'tuv':           return Race.TURKISH_ANGORA
      case 'autre':         return Race.OTHER
      case 'nsp' | 'nr':    raise MissingValueException("race", False)
      case _:       raise InvalidInputException('race', normalized, 'string representation')

  def to_neural_network_output(self):
    return list(type(self).__members__.values()).index(self)
class NumberOfCats(DefaultEnum):
  ONE = enum.auto()
  TWO = enum.auto()
  THREE = enum.auto()
  FOUR = enum.auto()
  AT_LEAST_FIVE = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'NumberOfCats':
    if raw is None:
      raise MissingValueException("number of cats")
    normalized = raw.lower()
    match normalized:
      case '1' | '2' | '3' | '4': return NumberOfCats(int(normalized))
      case 'plusde5':             return NumberOfCats.AT_LEAST_FIVE
      case _:                     raise InvalidInputException('number of cats', normalized, 'string representation')

  @staticmethod
  def from_int(number: int):
    if number <= 0: raise InvalidInputException('number of cats', number)
    if number < 5:  return number
    else:           return NumberOfCats.AT_LEAST_FIVE

class LivingSpace(DefaultEnum):
  APARTMENT_WITHOUT_BALCONY = enum.auto()
  APARTMENT_WITH_BALCONY = enum.auto()
  HOUSE_IN_A_SUBDIVISION = enum.auto()
  INDIVIDUAL_HOUSE = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'LivingSpace':
    if raw is None:
      raise MissingValueException("living space")
    normalized = raw.lower()
    match normalized:
      case 'asb': return LivingSpace.APARTMENT_WITHOUT_BALCONY
      case 'aab': return LivingSpace.APARTMENT_WITH_BALCONY
      case 'ml':  return LivingSpace.HOUSE_IN_A_SUBDIVISION
      case 'mi':  return LivingSpace.INDIVIDUAL_HOUSE
      case _:     raise InvalidInputException('living space', normalized, 'string representation')

  def to_neural_network_input(self) -> [float]:
    one_hot_encoding = [float(e == self) for e in type(self).__members__.values()]
    # type(self).__members__ is an ordered dictionary, so its elements will always be in the same order
    return one_hot_encoding

class Zone(DefaultEnum):
  URBAN = enum.auto()
  PERIURBAN = enum.auto()
  RURAL = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'Zone':
    if raw is None:
      raise MissingValueException("zone")
    normalized = raw.lower()
    match normalized:
      case 'u':   return Zone.URBAN
      case 'pu':  return Zone.PERIURBAN
      case 'r':   return Zone.RURAL
      case _:     raise InvalidInputException('zone', normalized, 'string representation')

class AbundanceOfNaturalAreas(DefaultEnum):
  DON_T_KNOW = enum.auto()
  LOW = enum.auto()
  MODERATE = enum.auto()
  HIGH = enum.auto()

  @staticmethod
  def from_string(raw: str) -> 'AbundanceOfNaturalAreas':
    if raw is None:
      raise MissingValueException("abundance of natural areas")
    normalized = raw.lower()
    match normalized:
      case '1' | '2' | '3': return AbundanceOfNaturalAreas(int(normalized))
      case 'nsp':           return AbundanceOfNaturalAreas.DON_T_KNOW
      case _:               raise InvalidInputException('abundance of natural areas', normalized, 'string representation')

class Cat(object):
  def __init__(self, attr_dict):
    # I don't see a reason why attributes using the function defined below should have their own enums classes yet,
    # since they're already in int representation in the dataset
    def convert_to_appropriate_int(attribute: str, upper_bound: int, source: str) -> int:
      if source is None:
        raise MissingValueException(attribute)
      try:
        number = int(source)
      except ValueError:
        raise InvalidInputException(attribute, source, 'string representation (must be an integer)')

      if not 1 <= number <= upper_bound:
        raise InputOutOfBoundsException(attribute, number, (1, upper_bound))

      return number

    self.timestamp = attr_dict.pop('timestamp')
    try:
      self.gender = Gender.from_string(attr_dict.pop('gender'))
    except MissingValueException:
      self.gender = random.choice([Gender.MALE, Gender.FEMALE])
    self.age = Age.from_string(attr_dict.pop('age'))
    self.race = attr_dict.pop('race') # already is a race object
    if attr_dict['number of cats'] == '5': attr_dict['number of cats'] = 'plusde5'
    self.number_of_cats = NumberOfCats.from_string(attr_dict.pop('number of cats'))
    self.living_space = LivingSpace.from_string(attr_dict.pop('living space'))
    self.zone = Zone.from_string(attr_dict.pop('zone'))
    self.abundance_of_natural_areas = AbundanceOfNaturalAreas.from_string(attr_dict.pop('abundance of natural areas'))


    for name, value in attr_dict.items():
      upper_bound = 5
      if name in ("bonding time"):
        upper_bound = 4
      match name:
        case "time spent outdoors" | "bonding time" | "bird capture frequency" | "small mammals capture frequency":
          if value == 0: value = '1'
      value = convert_to_appropriate_int(name, upper_bound, value)

      setattr(self, name.replace(" ", "_"), value)

  def to_neural_network_input_output_pair(self) -> tuple[np.ndarray, int | None]:
    arr = list()
    for k, v in vars(self).items():
      if k in ["timestamp", "race"]: continue
      if issubclass(type(v), DefaultEnum): arr += v.to_neural_network_input()
      else: arr += [(v - 1.0) / ((4 - 1) if k == "bonding_time" else (5 - 1))]
    return np.array(arr), self.race.to_neural_network_output() if self.race is not None else None

  def __eq__(self, other: 'Cat'):
    self_vars = vars(self)
    other_vars = vars(other)
    keys_to_exclude = ['timestamp', 'race']
    included_keys = (k for k in self_vars.keys() if k not in keys_to_exclude)

    return all(self_vars[key] == other_vars[key] for key in included_keys)