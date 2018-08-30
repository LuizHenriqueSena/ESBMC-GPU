#include <stdio.h>

int ti_clz(int a) {
	int i = 31;
	int result = 0;
	for(i = 31; i >= 0; i--) {
		if(!((a>>i)&0x01)) {
			result += 1;
		} else {
			return result;
		}
	}
	return 31;
}

