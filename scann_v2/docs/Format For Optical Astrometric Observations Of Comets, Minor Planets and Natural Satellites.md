Format For Optical Astrometric Observations Of Comets, Minor Planets and Natural Satellites
Astrometric observations of comets, minor planets and natural satellites submitted for publication in the Minor Planet Circulars (MPCs) and Minor Planet Electronic Circulars are represented by a standard 80-column record.
The formats are described below. The format beyond column 13 are identical for all three types of object (comets, minor planets and natural satellites). All observations must have a designation--never leave columns 1 to 12 blank.

SUMMARY OF FORMAT:
Please note that TABs must NOT be used. Columns marked as `blank' must contain spaces (ASCII 32). The Fortran formats listed below are for writing purposes.


MINOR PLANETS
   Columns     Format   Use
    1 -  5       A5     Packed minor planet number
    6 - 12       A7     Packed provisional designation, or a temporary designation
   13            A1     Discovery asterisk
Minor planet numbers and provisional designations are official designations assigned by the Minor Planet Center. Temporary designations are designations, preferably no more than six (6) characters long (the absolute maximum is seven (7) characters), assigned by the observer for new or unidentified objects. Temporary designations must consist of alphanumeric characters only: do not include spaces. All observations of the same "new" object reported in the same message must have the same temporary designation.


COMETS
   Columns     Format   Use
    1 -  4       I4     Periodic comet number
    5            A1     Letter indicating type of orbit
    6 - 12       A7     Provisional or temporary designation
   13            X      Not used, must be blank
Periodic comet numbers and provisional designations are official designations assigned by, respectively, the Minor Planet Center and Central Bureau for Astronomical Telegrams. Temporary designations are designations, up to six (6) characters long, assigned by the observer for new or unidentified objects. In practice, temporary designations on comet observations will be very rare.


NATURAL SATELLITES
   Columns     Format   Use
    1            A1     Planet identifier [Only if numbered]
    2 -  4       I3     Satellite number  [Only if numbered]
    5            A1     "S"
    6 - 12       A7     Provisional or temporary designation [Only if not numbered, see detailed notes below]
   13            X      Not used, must be blank

MINOR PLANETS, COMETS AND NATURAL SATELLITES
   Columns     Format   Use
   14            A1     Note 1
   15            A1     Note 2
   16 - 32              Date of observation
   33 - 44              Observed RA (J2000.0)
   45 - 56              Observed Decl. (J2000.0)
   57 - 65       9X     Must be blank
   66 - 71    F5.2,A1   Observed magnitude and band
                           (or nuclear/total flag for comets)
   72 - 77       X      Must be blank
   78 - 80       A3     Observatory code
DETAILED NOTES:

MINOR PLANETS

NUMBER
Columns 1-5 contain a zero-padded, right-justified number--e.g., an observation of (1) would be given as 00001, an observation of (3202) would be 03202. If there is no number these columns must be blank. Six-digit numbers are to be stored in packed form (A0000 = 100000), in order to be consistent with the format specifier earlier in this document.
PROVISIONAL/TEMPORARY DESIGNATION
Columns 6-12 contain the provisional designation or the temporary designation. The provisional designation is stored in a 7-character packed form.
Temporary designations are designations assigned by the observer for new or unidentified objects. Such designations must begin in column 6, should not exceed 6 characters in length, and should start with one or more letters.

It is important that every observation has a designation and that the same designation is used for all observations of the same object.

DISCOVERY ASTERISK
Discovery observations for new (or unidentified) objects should contain `*' in column 13. Only one asterisked observation per object is expected. Some objects consist of multiple designations, in that case each designation keeps its original discovery asterisk.

COMETS

PERIODIC COMET NUMBER
Periodic comets that have been observed at more than one return are assigned numbers. Reference should be made to the editorial notices on MPC 23803-23804 and 24421 for more complete details of the circumstances under which numbers are assigned.
Examples:

      Comet                  P/ Number    Columns 1-4
                                          will contain
      P/Halley                  1P          0001
      P/Encke                   2P          0002
      P/Biela                   3D          0003
      P/Wild 4                116P          0116
See the complete list of periodic comet numbers.
ORBIT TYPE
Column 5 contains `C' for a long-period comet, `P' for a short-period comet, `D' for a `defunct' comet, `X' for an uncertain comet, 'I' for an interstellar object, or `A' for a minor planet given a cometary designation or objects suspected to be comets.
PROVISIONAL DESIGNATION
Columns 6-12 contain a packed version of the provisional designation. The first two digits of the year are packed into a single character in column 6 (I = 18, J = 19, K = 20). Columns 7-8 contain the last two digits of the year. Column 9 contains the half-month letter. Columns 10-11 contain the order within the half-month. Column 12 will be normally be `0', except for split comets, when the fragment designation is stored there as a lower-case letter. Columns 6-12 may contain a minor-planet provisional designation. In such a situation column 12 will contain a capital letter.
   Examples:
   1995 A1   = J95A010
   1994 P1-B = J94P01b   refers to fragment B of 1994 P1
   1994 P1   = J94P010   refers to the whole comet 1994 P1

NATURAL SATELLITES

PLANET IDENTIFIER
A single character to represent the planet that the satellites belongs to. This is given in column 1 for those objects with Roman numeral designations and column 9 for those with provisional designations.
   Char   Planet
     J    Jupiter
     S    Saturn
     U    Uranus
     N    Neptune
SATELLITE NUMBER
For those objects with Roman numeral designations, columns 2-4 contain the number of the satellite.
COLUMN 5
Column 5 is always "S" for a satellite observation.
PROVISIONAL DESIGNATION
Columns 6-12 contain a packed version of the provisional designation for those objects without Roman numeral designations.
The first two digits of the year are packed into a single character in column 6 (I = 18, J = 19, K = 20). Columns 7-8 contain the last two digits of the year. Columns 10-11 contain the order within the year. Column 12 will be always be `0'. This is similar to the scheme used for comets.

   Examples
   123456789012
   J013S         Jupiter XIII
   N002S         Neptune II
       SJ99U030  S/1999 U 3    (Third new Uranian satellite discovered in 1999)
       SK20J010  S/2020 J 1    (First new Jovian satellite discovered in 2020)

COMETS, MINOR PLANETS AND NATURAL SATELLITES
NOTE 1
This column contains a alphabetical publishable note or (those sites that use program codes) an alphanumeric or non-alphanumeric character program code. The list of standard codes used for observations of minor planets is given in each batch of MPCs.
NOTE 2
This column serves two purposes. For those observations which have been converted to the J2000.0 system by rotating B1950.0 coordinates this column contains `A', to indicate that the value has been adjusted. For those observations reduced in the J2000.0 system this column is used to indicate how the observation was made. The following codes will be used: In addition, there are 'X' and 'x' which are used only for already-filed observations. 'X' was given originally only to discovery observations that were approximate or semi-accurate and that had accurate measures corresponding to the time of discovery: this has been extended to other replaced discovery observations. Observations marked 'X'/'x' are to be suppressed in residual blocks. They are retained so that there exists an original record of a discovery. These codes MUST NOT be used on observation submissions.
      P   Photographic
      e   Encoder
      C   CCD
      B   CMOS
      T   Meridian or transit circle
      M   Micrometer
     V/v  "Roving Observer" observation
     R/r  Radar observation
     S/s  Satellite observation
      c   Corrected-without-republication CCD observation [MUST NOT be used on observation submissions]
      E   Occultation-derived observations
      O   Offset observations (used only for observations of natural satellites)
      H   Hipparcos geocentric observations
      N   Normal place
      n   Mini-normal place derived from averaging observations from video frames

      D   CCD observation converted from original XML-formatted submission [MUST NOT be used on observation submissions]
      Z   Photographic observation converted from original XML-formatted submission [MUST NOT be used on observation submissions]
     W/w  "Roving observer" observation converted from original XML-formatted submission [MUST NOT be used on observation submissions]
     Q/q  Radar observation converted from original XML-formatted submission [MUST NOT be used on observation submissions]
     T/t  Satellite observation converted from original XML-formatted submission [MUST NOT be used on observation submissions]
DATE OF OBSERVATIONS
Columns 16-32 contain the date and UTC time, usually corresponding to the mid-point of observation. If the astrometry refers to one end of a trailed image, then the time of observation should be either the start time of the exposure or the finish time of the exposure, depending on which end of the trail was measured. The format is "YYYY MM DD.dddddd", with the decimal day of observation normally being given to a precision of 0.00001 days. Where such precision is justified, there is the option of recording times to 0.000001 days.
OBSERVED RA (J2000.0)
Columns 33-44 contain the observed J2000.0 right ascension. The format is "HH MM SS.ddd", with the seconds of R.A. normally being given to a precision of 0.01s. There is the option of recording the right ascension to 0.001s, where such precision is justified.
OBSERVED DECL (J2000.0)
Columns 45-56 contain the observed J2000.0 declination. The format is "sDD MM SS.dd" (with "s" being the sign), with the seconds of Decl. normally being given to a precision of 0.1". There is the option of recording the declination to 0".01, where such precision is justified.
OBSERVED MAGNITUDE AND BAND
The observed magnitude (normally to a precision of 0.1 mag.) and the band in which the measurement was made. The observed magnitude can be given to 0.01 mag., where such precision is justified. The default magnitude scale is photographic, although magnitudes may be given in V- or R-band, for example. In the past for comets, the magnitude was specified as being nuclear, N, or total, T, but observers are now encouraged to provide the actual band used.
The current list of acceptable magnitude bands is: B, V, R, I, J, W, U, C, L, H, K, Y, G, g, r, i, w, y, z, o, c, v, u. Non-recognized magnitude bands will cause observations to be rejected. Addition of new recognised bands requires knowledge of a standard correction to convert a magnitude in that band to V. Conversion to V band used by MPC is located here.

The formerly-used "C" band to indicate "clear" or "no filter" is no longer valid for newly-submitted observations, but will remain on previously-submitted observations.

OBSERVATORY CODE
Observatory codes are stored in columns 78-80. Lists of observatory codes are published from time to time in the MPCs. Note that new observatory codes are assigned only upon receipt of acceptable astrometric observations.

full example：
COD N89
CON X.Gao, Urumqi No.1 Senior High School,XinJiang,China [webmaster@xjltp.com]
OBS Y.Ou, X.Gao
MEA Y.Ou
TEL 0.5-m f/4 reflector + CCD
ACK MPCReport file updated 2026.02.09 14:05:42
AC2 webmaster@xjltp.com
NET UCAC-4
     0003I    C2025 12 20.85808 10 38 29.630+07 44 11.48         13.40V      N89
     16-01    C2025 11 16.54789 00 45 05.108+41 06 33.60         19.15V      N89
     16-02    C2025 11 16.84718 00 45 05.134+41 06 33.55         19.64V      N89
     17-01    C2025 11 17.68528 00 45 05.026+41 06 33.44         19.69V      N89
     18-01    C2025 11 18.56222 00 45 05.094+41 06 32.92         20.52V      N89
     18-02    C2025 11 18.81755 00 45 05.075+41 06 33.41         19.94V      N89
     19-01    C2025 11 19.67481 00 45 05.089+41 06 33.82         19.46V      N89
     19-02    C2025 11 19.83846 00 45 05.100+41 06 33.69         20.08V      N89
     20-01    C2025 11 20.53280 00 45 05.025+41 06 33.86         19.28V      N89
     20-02    C2025 11 20.80862 00 45 05.264+41 06 31.98         19.51V      N89
     21-01    C2025 11 21.53817 00 45 05.061+41 06 33.70         18.62V      N89
     21-02    C2025 11 21.85006 00 45 05.183+41 06 33.75         18.27V      N89
     23-02    C2025 11 23.85211 00 45 05.098+41 06 34.00         17.92V      N89
     25-01    C2025 11 25.57179 00 45 05.164+41 06 33.83         19.57V      N89
     PSN01    C2026 02 05.60163 02 05 49.500+38 12 25.56         18.98V      N89
     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      N89
     TLE02    C2026 01 26.83814 11 57 26.102+10 16 47.29         16.43V      N89
     TLE03?   C2025 07 18.74299 16 15 15.517+63 58 51.21         16.71V      N89
     TLE04    C2026 02 05.60163 02 05 20.329+38 31 28.82         18.15V      N89
     VAR01    C2026 02 08.67954 02 59 54.821+35 32 08.49         17.65V      N89
08068         C2026 02 07.73946 06 53 11.199+19 24 27.46         16.88V      N89
09424         C2026 01 30.64312 02 27 17.597+37 42 08.06         16.87V      N89
C0001         C2026 02 06.73331 07 12 55.919+23 06 25.16         15.22V      N89
C0002         C2026 02 06.73331 07 12 44.153+23 06 15.48         16.74V      N89
K1161         C2026 02 08.67757 02 58 51.503+41 48 09.10         18.78V      N89
----- end -----