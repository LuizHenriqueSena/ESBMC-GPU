#include "../stubs.h"
#include "../base.h"

#define MAXLINE BASE_SZ

int main (void)
{
  char fbuf[MAXLINE+1];
  int fb;
  int c1, c2, c3;

  fb = 0;
  while ((c1 = nondet_int ()) != EOF)
  {
    c2 = nondet_int ();
    if (c2 == EOF)
      break;

    c3 = nondet_int ();
    if (c3 == EOF)
      break;

    if (c1 == '=' || c2 == '=')
      continue;

    /* OK */
    fbuf[fb] = c1;

    /* OK */
    if (fbuf[fb] == '\n' || fb >= MAXLINE)
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* OK */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
      fb++;

    /* OK */
    fbuf[fb] = c2;

    /* OK */
    if (fbuf[fb] == '\n' || fb >= MAXLINE)
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* OK */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
      fb++;

    if (c3 == '=')
      continue;
    /* OK */
    fbuf[fb] = c3;

    /* OK */
    if (fbuf[fb] == '\n' || fb >= MAXLINE)
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* OK */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
      fb++;
  }

  /* force out partial last line */
  if (fb > 0)
  {
    /* OK */
    fbuf[fb] = EOS;
  }

  return 0;
}
