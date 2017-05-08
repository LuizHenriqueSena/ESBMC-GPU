#include "../wu-ftpd.h"

/* Allocated size of buffer pathname[] in main () */
#define PATHNAME_SZ MAXPATHLEN+1
#if 1
char *r_strcpy (char *dest, const char *src)
{
  int i;
  char tmp;
  for (i = 0; ; i++) {
    tmp = src[i];
    /* r_strcpy RELEVANT */
    dest[i] = tmp; // DO NOT CHANGE THE POSITION OF THIS LINE
    if (src[i] == EOS)
      break;
  }
  return dest;
}
#endif

char *
realpath(const char *pathname, char *result, char* chroot_path)
{
  char curpath[MAXPATHLEN];

  if (result == NULL)
    return(NULL);

  if(pathname == NULL){
    *result = EOS; 
    return(NULL);
  }

  /* BAD */
  r_strcpy(curpath, pathname);

  return result;
}

int main ()
{
  char pathname [PATHNAME_SZ];
  char result [MAXPATHLEN];
  char chroot_path [MAXPATHLEN];

  pathname [PATHNAME_SZ-1] = EOS;
  result [MAXPATHLEN-1] = EOS;
  chroot_path [MAXPATHLEN-1] = EOS;

  realpath(pathname, result, chroot_path);

  return 0;
}
