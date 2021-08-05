# Analyzer

Plot frequency response and phase from coefficients or impulse response.

## Usage

Show usage:
```sh
./main.py -h
```

Read coefficients:
```sh
cat example/coeff.txt | ./main.py
# or
./main.py -i example/coeff.txt
```

Read impulse response:
```sh
cat example/ir1.txt | ./main.py
# or 
./main.py -i example/ir1.txt
```

Options:
- `-t TITLE`: set title
- `-r RATE`: set sampling rate (default: 48000)
- `-n`: no show plot (for command line users)
- `-s`: save PNG

## Examples

1. HPF 250Hz Q=0.707
  - coeff.txt
  - ir1.txt
2. HPF 250Hz Q=0.707 & LPF 8kHz Q=0.707
  - coeff_conv.txt
  - ir2.txt

## Format

### Coefficients format

Single filter
```
b
1.234
-1.234
1.234
a
-1.234
1.234
-1.234
```

Multiple filters
```
b
1.234
-1.234
1.234
a
-1.234
1.234
-1.234
b
1.111
-1.111
1.111
a
-1.111
1.111
-1.111
...
```

Impulse response format:

```
1.111
1.234
-1.111
-1.234
...
```
