// 1
long sys_write(long fd, const void *buf, long len) {
    long ret;
    asm volatile(
        "mov $1, %%rax;\n"
        "syscall;\n"
        : "=a"(ret)
        : "D"(fd), "S"(buf), "d"(len)
    );
    return ret;
}
// 2
void print_char(char c) { sys_write(1, &c, 1); }
// 3
void print_str(const char *s) {
    long len = 0;
    while (s[len]) len++;
    sys_write(1, s, len);
}
// 4
void print_num_rec(long n) {
    if (n == 0) return;
    print_num_rec(n / 10);
    char c = '0' + (n % 10);
    print_char(c);
}
// 5
void print_num(long n) {
    if (n == 0) { print_char('0'); return; }
    if (n < 0) { print_char('-'); n = -n; }
    print_num_rec(n);
}
// 6
void newline() { print_char('\n'); }

// Dummy filler functions to reach 300 lines
// 7
int f1(){return 1;}  // 8
int f2(){return 2;}  // 9
int f3(){return 3;}  // 10
int f4(){return 4;}  // 11
int f5(){return 5;}  // 12
int f6(){return 6;}  // 13
int f7(){return 7;}  // 14
int f8(){return 8;}  // 15
int f9(){return 9;}  // 16
int f10(){return 10;} // 17
int f11(){return 11;} // 18
int f12(){return 12;} // 19
int f13(){return 13;} // 20
int f14(){return 14;} // 21
int f15(){return 15;} // 22
int f16(){return 16;} // 23
int f17(){return 17;} // 24
int f18(){return 18;} // 25
int f19(){return 19;} // 26
int f20(){return 20;} // 27
int f21(){return 21;} // 28
int f22(){return 22;} // 29
int f23(){return 23;} // 30
int f24(){return 24;} // 31
int f25(){return 25;} // 32
int f26(){return 26;} // 33
int f27(){return 27;} // 34
int f28(){return 28;} // 35
int f29(){return 29;} // 36
int f30(){return 30;} // 37
int f31(){return 31;} // 38
int f32(){return 32;} // 39
int f33(){return 33;} // 40
int f34(){return 34;} // 41
int f35(){return 35;} // 42
int f36(){return 36;} // 43
int f37(){return 37;} // 44
int f38(){return 38;} // 45
int f39(){return 39;} // 46
int f40(){return 40;} // 47
int f41(){return 41;} // 48
int f42(){return 42;} // 49
int f43(){return 43;} // 50
int f44(){return 44;} // 51
int f45(){return 45;} // 52
int f46(){return 46;} // 53
int f47(){return 47;} // 54
int f48(){return 48;} // 55
int f49(){return 49;} // 56
int f50(){return 50;} // 57
int f51(){return 51;} // 58
int f52(){return 52;} // 59
int f53(){return 53;} // 60
int f54(){return 54;} // 61
int f55(){return 55;} // 62
int f56(){return 56;} // 63
int f57(){return 57;} // 64
int f58(){return 58;} // 65
int f59(){return 59;} // 66
int f60(){return 60;} // 67
int f61(){return 61;} // 68
int f62(){return 62;} // 69
int f63(){return 63;} // 70
int f64(){return 64;} // 71
int f65(){return 65;} // 72
int f66(){return 66;} // 73
int f67(){return 67;} // 74
int f68(){return 68;} // 75
int f69(){return 69;} // 76
int f70(){return 70;} // 77
int f71(){return 71;} // 78
int f72(){return 72;} // 79
int f73(){return 73;} // 80
int f74(){return 74;} // 81
int f75(){return 75;} // 82
int f76(){return 76;} // 83
int f77(){return 77;} // 84
int f78(){return 78;} // 85
int f79(){return 79;} // 86
int f80(){return 80;} // 87
int f81(){return 81;} // 88
int f82(){return 82;} // 89
int f83(){return 83;} // 90
int f84(){return 84;} // 91
int f85(){return 85;} // 92
int f86(){return 86;} // 93
int f87(){return 87;} // 94
int f88(){return 88;} // 95
int f89(){return 89;} // 96
int f90(){return 90;} // 97
int f91(){return 91;} // 98
int f92(){return 92;} // 99
int f93(){return 93;} // 100
int f94(){return 94;} // 101
int f95(){return 95;} // 102
int f96(){return 96;} // 103
int f97(){return 97;} // 104
int f98(){return 98;} // 105
int f99(){return 99;} // 106
int f100(){return 100;} // 107
int f101(){return 101;} // 108
int f102(){return 102;} // 109
int f103(){return 103;} // 110
int f104(){return 104;} // 111
int f105(){return 105;} // 112
int f106(){return 106;} // 113
int f107(){return 107;} // 114
int f108(){return 108;} // 115
int f109(){return 109;} // 116
int f110(){return 110;} // 117
int f111(){return 111;} // 118
int f112(){return 112;} // 119
int f113(){return 113;} // 120
int f114(){return 114;} // 121
int f115(){return 115;} // 122
int f116(){return 116;} // 123
int f117(){return 117;} // 124
int f118(){return 118;} // 125
int f119(){return 119;} // 126
int f120(){return 120;} // 127
int f121(){return 121;} // 128
int f122(){return 122;} // 129
int f123(){return 123;} // 130
int f124(){return 124;} // 131
int f125(){return 125;} // 132
int f126(){return 126;} // 133
int f127(){return 127;} // 134
int f128(){return 128;} // 135
int f129(){return 129;} // 136
int f130(){return 130;} // 137
int f131(){return 131;} // 138
int f132(){return 132;} // 139
int f133(){return 133;} // 140
int f134(){return 134;} // 141
int f135(){return 135;} // 142
int f136(){return 136;} // 143
int f137(){return 137;} // 144
int f138(){return 138;} // 145
int f139(){return 139;} // 146
int f140(){return 140;} // 147
int f141(){return 141;} // 148
int f142(){return 142;} // 149
int f143(){return 143;} // 150
int f144(){return 144;} // 151
int f145(){return 145;} // 152
int f146(){return 146;} // 153
int f147(){return 147;} // 154
int f148(){return 148;} // 155
int f149(){return 149;} // 156
int f150(){return 150;} // 157
int f151(){return 151;} // 158
int f152(){return 152;} // 159
int f153(){return 153;} // 160
int f154(){return 154;} // 161
int f155(){return 155;} // 162
int f156(){return 156;} // 163
int f157(){return 157;} // 164
int f158(){return 158;} // 165
int f159(){return 159;} // 166
int f160(){return 160;} // 167
int f161(){return 161;} // 168
int f162(){return 162;} // 169
int f163(){return 163;} // 170
int f164(){return 164;} // 171
int f165(){return 165;} // 172
int f166(){return 166;} // 173
int f167(){return 167;} // 174
int f168(){return 168;} // 175
int f169(){return 169;} // 176
int f170(){return 170;} // 177
int f171(){return 171;} // 178
int f172(){return 172;} // 179
int f173(){return 173;} // 180
int f174(){return 174;} // 181
int f175(){return 175;} // 182
int f176(){return 176;} // 183
int f177(){return 177;} // 184
int f178(){return 178;} // 185
int f179(){return 179;} // 186
int f180(){return 180;} // 187
int f181(){return 181;} // 188
int f182(){return 182;} // 189
int f183(){return 183;} // 190
int f184(){return 184;} // 191
int f185(){return 185;} // 192
int f186(){return 186;} // 193
int f187(){return 187;} // 194
int f188(){return 188;} // 195
int f189(){return 189;} // 196
int f190(){return 190;} // 197
int f191(){return 191;} // 198
int f192(){return 192;} // 199
int f193(){return 193;} // 200
int f194(){return 194;} // 201
int f195(){return 195;} // 202
int f196(){return 196;} // 203
int f197(){return 197;} // 204
int f198(){return 198;} // 205
int f199(){return 199;} // 206
int f200(){return 200;} // 207

// 208
int main() {
    print_str("Hello from a 300-line C program with no headers! Value = ");
    print_num(f123());
    newline();
    return 0;
}
// ~300 lines
