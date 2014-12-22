-- haskell version of loadhitran module
import Data.List.Split
module loadhitran
( importhitran
, filterhitran
, processhitran
) where
-- extract all data from HITRAN-type file
importhitran :: Str -> Hraw
importhitran a = splitPlaces hitPlaces 



