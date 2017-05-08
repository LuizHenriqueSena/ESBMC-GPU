/*
 * This one just blindly copies the input into buffer and writes '>''\0' at the
 * end.
 */

#include "../stubs.h"

int main (void)
{
  char buffer[BASE_SZ+1];
  char input[BASE_SZ+70];
  char *buf;
  char *buflim;
  char *in;
  char cur;

//  shouldn't be necessary unless checking for safety of *in
//  input[BASE_SZ+70-1] = EOS;
  in = input;
  buf = buffer;
  buflim = &buffer[sizeof buffer - 1];
    // didn't reserve enough space for both '>' and '\0'!

  cur = *in;
  while (cur != EOS)
  {
    if (buf == buflim)
      break;

    *buf = cur;
    buf++;
out:
    in++;
    cur = *in;
  }

  *buf = '>';
  buf++;

  /* BAD */
  *buf = EOS;
  buf++;

  return 0;
}
