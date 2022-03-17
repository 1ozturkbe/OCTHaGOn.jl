$offlisting
*  MINLP written by GAMS Convert at 12/19/05 12:07:04
*  
*  Equation counts
*      Total        E        G        L        N        X        C
*        348        2        2      344        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*        145       89        0       56        0        0        0        0
*  FX      0        0        0        0        0        0        0        0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*       1381     1365       16        0
*
*  Solve m using MINLP minimizing objvar;


Variables  i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19
          ,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36
          ,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53
          ,i54,i55,i56,objvar,x58,x59,x60,x61,x62,x63,x64,x65,x66,x67,x68,x69
          ,x70,x71,x72,x73,x74,x75,x76,x77,x78,x79,x80,x81,x82,x83,x84,x85,x86
          ,x87,x88,x89,x90,x91,x92,x93,x94,x95,x96,x97,x98,x99,x100,x101,x102
          ,x103,x104,x105,x106,x107,x108,x109,x110,x111,x112,x113,x114,x115
          ,x116,x117,x118,x119,x120,x121,x122,x123,x124,x125,x126,x127,x128
          ,x129,x130,x131,x132,x133,x134,x135,x136,x137,x138,x139,x140,x141
          ,x142,x143,x144,x145;

Integer Variables  i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17
          ,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34
          ,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51
          ,i52,i53,i54,i55,i56;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,e51,e52,e53
          ,e54,e55,e56,e57,e58,e59,e60,e61,e62,e63,e64,e65,e66,e67,e68,e69,e70
          ,e71,e72,e73,e74,e75,e76,e77,e78,e79,e80,e81,e82,e83,e84,e85,e86,e87
          ,e88,e89,e90,e91,e92,e93,e94,e95,e96,e97,e98,e99,e100,e101,e102,e103
          ,e104,e105,e106,e107,e108,e109,e110,e111,e112,e113,e114,e115,e116
          ,e117,e118,e119,e120,e121,e122,e123,e124,e125,e126,e127,e128,e129
          ,e130,e131,e132,e133,e134,e135,e136,e137,e138,e139,e140,e141,e142
          ,e143,e144,e145,e146,e147,e148,e149,e150,e151,e152,e153,e154,e155
          ,e156,e157,e158,e159,e160,e161,e162,e163,e164,e165,e166,e167,e168
          ,e169,e170,e171,e172,e173,e174,e175,e176,e177,e178,e179,e180,e181
          ,e182,e183,e184,e185,e186,e187,e188,e189,e190,e191,e192,e193,e194
          ,e195,e196,e197,e198,e199,e200,e201,e202,e203,e204,e205,e206,e207
          ,e208,e209,e210,e211,e212,e213,e214,e215,e216,e217,e218,e219,e220
          ,e221,e222,e223,e224,e225,e226,e227,e228,e229,e230,e231,e232,e233
          ,e234,e235,e236,e237,e238,e239,e240,e241,e242,e243,e244,e245,e246
          ,e247,e248,e249,e250,e251,e252,e253,e254,e255,e256,e257,e258,e259
          ,e260,e261,e262,e263,e264,e265,e266,e267,e268,e269,e270,e271,e272
          ,e273,e274,e275,e276,e277,e278,e279,e280,e281,e282,e283,e284,e285
          ,e286,e287,e288,e289,e290,e291,e292,e293,e294,e295,e296,e297,e298
          ,e299,e300,e301,e302,e303,e304,e305,e306,e307,e308,e309,e310,e311
          ,e312,e313,e314,e315,e316,e317,e318,e319,e320,e321,e322,e323,e324
          ,e325,e326,e327,e328,e329,e330,e331,e332,e333,e334,e335,e336,e337
          ,e338,e339,e340,e341,e342,e343,e344,e345,e346,e347,e348;


e1..    objvar - x58 - x59 - x60 - x61 - x62 - x63 - x64 - x65 - x66 - x67
      - x68 - x69 - x70 - x71 =E= 0;

e2..    x72 - x73 =G= 0;

e3..    x74 - x75 =G= 0;

e4..    i1 - i2 =E= 0;

e5..  - 11.31*i1 - x58 + 0.5*x76 + 0.5*x77 =L= 0;

e6..    13*i1 - x59 + 0.5*x78 + 0.5*x79 =L= 13;

e7..  - 11.31*i3 + 0.5*x76 + 0.5*x80 - x81 =L= 0;

e8..    13*i3 + 0.5*x78 + 0.5*x82 - x83 =L= 13;

e9..  - 11.31*i5 + 0.5*x76 + 0.5*x84 - x85 =L= 0;

e10..    13*i5 + 0.5*x78 + 0.5*x86 - x87 =L= 13;

e11..  - 11.31*i7 + 0.5*x76 + 0.5*x88 - x89 =L= 0;

e12..    13*i7 + 0.5*x78 + 0.5*x90 - x91 =L= 13;

e13..  - 11.31*i9 + 0.5*x76 + 0.5*x92 - x93 =L= 0;

e14..    13*i9 + 0.5*x78 + 0.5*x94 - x95 =L= 13;

e15..  - 11.31*i11 + 0.5*x76 + 0.5*x96 - x97 =L= 0;

e16..    13*i11 + 0.5*x78 + 0.5*x98 - x99 =L= 13;

e17..  - 11.31*i13 + 0.5*x76 + 0.5*x100 - x101 =L= 0;

e18..    13*i13 + 0.5*x78 + 0.5*x102 - x103 =L= 13;

e19..  - 11.31*i15 - x60 + 0.5*x77 + 0.5*x80 =L= 0;

e20..    13*i15 - x61 + 0.5*x79 + 0.5*x82 =L= 13;

e21..  - 11.31*i17 + 0.5*x77 + 0.5*x84 - x104 =L= 0;

e22..    13*i17 + 0.5*x79 + 0.5*x86 - x105 =L= 13;

e23..  - 11.31*i19 + 0.5*x77 + 0.5*x88 - x106 =L= 0;

e24..    13*i19 + 0.5*x79 + 0.5*x90 - x107 =L= 13;

e25..  - 11.31*i21 + 0.5*x77 + 0.5*x92 - x108 =L= 0;

e26..    13*i21 + 0.5*x79 + 0.5*x94 - x109 =L= 13;

e27..  - 11.31*i23 + 0.5*x77 + 0.5*x96 - x110 =L= 0;

e28..    13*i23 + 0.5*x79 + 0.5*x98 - x111 =L= 13;

e29..  - 11.31*i25 + 0.5*x77 + 0.5*x100 - x112 =L= 0;

e30..    13*i25 + 0.5*x79 + 0.5*x102 - x113 =L= 13;

e31..  - 11.31*i27 - x62 + 0.5*x80 + 0.5*x84 =L= 0;

e32..    13*i27 - x63 + 0.5*x82 + 0.5*x86 =L= 13;

e33..  - 11.31*i29 + 0.5*x80 + 0.5*x88 - x114 =L= 0;

e34..    13*i29 + 0.5*x82 + 0.5*x90 - x115 =L= 13;

e35..  - 11.31*i31 + 0.5*x80 + 0.5*x92 - x116 =L= 0;

e36..    13*i31 + 0.5*x82 + 0.5*x94 - x117 =L= 13;

e37..  - 11.31*i33 + 0.5*x80 + 0.5*x96 - x118 =L= 0;

e38..    13*i33 + 0.5*x82 + 0.5*x98 - x119 =L= 13;

e39..  - 11.31*i35 + 0.5*x80 + 0.5*x100 - x120 =L= 0;

e40..    13*i35 + 0.5*x82 + 0.5*x102 - x121 =L= 13;

e41..  - 11.31*i37 - x64 + 0.5*x84 + 0.5*x88 =L= 0;

e42..    13*i37 - x65 + 0.5*x86 + 0.5*x90 =L= 13;

e43..  - 11.31*i39 + 0.5*x84 + 0.5*x92 - x122 =L= 0;

e44..    13*i39 + 0.5*x86 + 0.5*x94 - x123 =L= 13;

e45..  - 11.31*i41 + 0.5*x84 + 0.5*x96 - x124 =L= 0;

e46..    13*i41 + 0.5*x86 + 0.5*x98 - x125 =L= 13;

e47..  - 11.31*i43 + 0.5*x84 + 0.5*x100 - x126 =L= 0;

e48..    13*i43 + 0.5*x86 + 0.5*x102 - x127 =L= 13;

e49..  - 11.31*i45 - x66 + 0.5*x88 + 0.5*x92 =L= 0;

e50..    13*i45 - x67 + 0.5*x90 + 0.5*x94 =L= 13;

e51..  - 11.31*i47 + 0.5*x88 + 0.5*x96 - x128 =L= 0;

e52..    13*i47 + 0.5*x90 + 0.5*x98 - x129 =L= 13;

e53..  - 11.31*i49 + 0.5*x88 + 0.5*x100 - x130 =L= 0;

e54..    13*i49 + 0.5*x90 + 0.5*x102 - x131 =L= 13;

e55..  - 11.31*i51 - x68 + 0.5*x92 + 0.5*x96 =L= 0;

e56..    13*i51 - x69 + 0.5*x94 + 0.5*x98 =L= 13;

e57..  - 11.31*i53 + 0.5*x92 + 0.5*x100 - x132 =L= 0;

e58..    13*i53 + 0.5*x94 + 0.5*x102 - x133 =L= 13;

e59..  - 11.31*i55 - x70 + 0.5*x96 + 0.5*x100 =L= 0;

e60..    13*i55 - x71 + 0.5*x98 + 0.5*x102 =L= 13;

e61..  - 0.353557*x76 - 0.176775*x78 =L= -2;

e62..  - 0.176775*x76 - 0.353556*x78 =L= -2;

e63..  - 0.353557*x77 - 0.176775*x79 =L= -2;

e64..  - 0.176775*x77 - 0.353556*x79 =L= -2;

e65..  - 0.353557*x80 - 0.176775*x82 =L= -2;

e66..  - 0.176775*x80 - 0.353556*x82 =L= -2;

e67..  - 0.235705*x84 - 0.11785*x86 =L= -2;

e68..  - 0.117851*x84 - 0.235703*x86 =L= -2;

e69..  - 0.235705*x88 - 0.11785*x90 =L= -2;

e70..  - 0.117851*x88 - 0.235703*x90 =L= -2;

e71..  - 0.471409*x92 - 0.2357*x94 =L= -2;

e72..  - 0.235705*x92 - 0.4714*x94 =L= -2;

e73..  - 0.471409*x96 - 0.2357*x98 =L= -2;

e74..  - 0.235705*x96 - 0.4714*x98 =L= -2;

e75..  - 0.471409*x100 - 0.2357*x102 =L= -2;

e76..  - 0.235705*x100 - 0.4714*x102 =L= -2;

e77..    x72 + 0.5*x76 =L= 11.31;

e78..  - x72 + 0.5*x76 =L= 0;

e79..    x75 + 0.5*x78 =L= 13;

e80..  - x75 + 0.5*x78 =L= 0;

e81..    x73 + 0.5*x77 =L= 11.31;

e82..  - x73 + 0.5*x77 =L= 0;

e83..    x74 + 0.5*x79 =L= 13;

e84..  - x74 + 0.5*x79 =L= 0;

e85..    0.5*x80 + x134 =L= 11.31;

e86..    0.5*x80 - x134 =L= 0;

e87..    0.5*x82 + x135 =L= 13;

e88..    0.5*x82 - x135 =L= 0;

e89..    0.5*x84 + x136 =L= 11.31;

e90..    0.5*x84 - x136 =L= 0;

e91..    0.5*x86 + x137 =L= 13;

e92..    0.5*x86 - x137 =L= 0;

e93..    0.5*x88 + x138 =L= 11.31;

e94..    0.5*x88 - x138 =L= 0;

e95..    0.5*x90 + x139 =L= 13;

e96..    0.5*x90 - x139 =L= 0;

e97..    0.5*x92 + x140 =L= 11.31;

e98..    0.5*x92 - x140 =L= 0;

e99..    0.5*x94 + x141 =L= 13;

e100..    0.5*x94 - x141 =L= 0;

e101..    0.5*x96 + x142 =L= 11.31;

e102..    0.5*x96 - x142 =L= 0;

e103..    0.5*x98 + x143 =L= 13;

e104..    0.5*x98 - x143 =L= 0;

e105..    0.5*x100 + x144 =L= 11.31;

e106..    0.5*x100 - x144 =L= 0;

e107..    0.5*x102 + x145 =L= 13;

e108..    0.5*x102 - x145 =L= 0;

e109..  - x58 + x72 - x73 =L= 0;

e110..  - x58 - x72 + x73 =L= 0;

e111..  - x59 - x74 + x75 =L= 0;

e112..  - x59 + x74 - x75 =L= 0;

e113..  - 11.31*i1 - 11.31*i2 - x72 + x73 + 0.5*x76 + 0.5*x77 =L= 0;

e114..  - 11.31*i1 + 11.31*i2 + x72 - x73 + 0.5*x76 + 0.5*x77 =L= 11.31;

e115..    13*i1 - 13*i2 + x74 - x75 + 0.5*x78 + 0.5*x79 =L= 13;

e116..    13*i1 + 13*i2 - x74 + x75 + 0.5*x78 + 0.5*x79 =L= 26;

e117..    x72 - x81 - x134 =L= 0;

e118..  - x72 - x81 + x134 =L= 0;

e119..    x75 - x83 - x135 =L= 0;

e120..  - x75 - x83 + x135 =L= 0;

e121..  - 11.31*i3 - 11.31*i4 - x72 + 0.5*x76 + 0.5*x80 + x134 =L= 0;

e122..  - 11.31*i3 + 11.31*i4 + x72 + 0.5*x76 + 0.5*x80 - x134 =L= 11.31;

e123..    13*i3 - 13*i4 - x75 + 0.5*x78 + 0.5*x82 + x135 =L= 13;

e124..    13*i3 + 13*i4 + x75 + 0.5*x78 + 0.5*x82 - x135 =L= 26;

e125..    x72 - x85 - x136 =L= 0;

e126..  - x72 - x85 + x136 =L= 0;

e127..    x75 - x87 - x137 =L= 0;

e128..  - x75 - x87 + x137 =L= 0;

e129..  - 11.31*i5 - 11.31*i6 - x72 + 0.5*x76 + 0.5*x84 + x136 =L= 0;

e130..  - 11.31*i5 + 11.31*i6 + x72 + 0.5*x76 + 0.5*x84 - x136 =L= 11.31;

e131..    13*i5 - 13*i6 - x75 + 0.5*x78 + 0.5*x86 + x137 =L= 13;

e132..    13*i5 + 13*i6 + x75 + 0.5*x78 + 0.5*x86 - x137 =L= 26;

e133..    x72 - x89 - x138 =L= 0;

e134..  - x72 - x89 + x138 =L= 0;

e135..    x75 - x91 - x139 =L= 0;

e136..  - x75 - x91 + x139 =L= 0;

e137..  - 11.31*i7 - 11.31*i8 - x72 + 0.5*x76 + 0.5*x88 + x138 =L= 0;

e138..  - 11.31*i7 + 11.31*i8 + x72 + 0.5*x76 + 0.5*x88 - x138 =L= 11.31;

e139..    13*i7 - 13*i8 - x75 + 0.5*x78 + 0.5*x90 + x139 =L= 13;

e140..    13*i7 + 13*i8 + x75 + 0.5*x78 + 0.5*x90 - x139 =L= 26;

e141..    x72 - x93 - x140 =L= 0;

e142..  - x72 - x93 + x140 =L= 0;

e143..    x75 - x95 - x141 =L= 0;

e144..  - x75 - x95 + x141 =L= 0;

e145..  - 11.31*i9 - 11.31*i10 - x72 + 0.5*x76 + 0.5*x92 + x140 =L= 0;

e146..  - 11.31*i9 + 11.31*i10 + x72 + 0.5*x76 + 0.5*x92 - x140 =L= 11.31;

e147..    13*i9 - 13*i10 - x75 + 0.5*x78 + 0.5*x94 + x141 =L= 13;

e148..    13*i9 + 13*i10 + x75 + 0.5*x78 + 0.5*x94 - x141 =L= 26;

e149..    x72 - x97 - x142 =L= 0;

e150..  - x72 - x97 + x142 =L= 0;

e151..    x75 - x99 - x143 =L= 0;

e152..  - x75 - x99 + x143 =L= 0;

e153..  - 11.31*i11 - 11.31*i12 - x72 + 0.5*x76 + 0.5*x96 + x142 =L= 0;

e154..  - 11.31*i11 + 11.31*i12 + x72 + 0.5*x76 + 0.5*x96 - x142 =L= 11.31;

e155..    13*i11 - 13*i12 - x75 + 0.5*x78 + 0.5*x98 + x143 =L= 13;

e156..    13*i11 + 13*i12 + x75 + 0.5*x78 + 0.5*x98 - x143 =L= 26;

e157..    x72 - x101 - x144 =L= 0;

e158..  - x72 - x101 + x144 =L= 0;

e159..    x75 - x103 - x145 =L= 0;

e160..  - x75 - x103 + x145 =L= 0;

e161..  - 11.31*i13 - 11.31*i14 - x72 + 0.5*x76 + 0.5*x100 + x144 =L= 0;

e162..  - 11.31*i13 + 11.31*i14 + x72 + 0.5*x76 + 0.5*x100 - x144 =L= 11.31;

e163..    13*i13 - 13*i14 - x75 + 0.5*x78 + 0.5*x102 + x145 =L= 13;

e164..    13*i13 + 13*i14 + x75 + 0.5*x78 + 0.5*x102 - x145 =L= 26;

e165..  - x60 + x73 - x134 =L= 0;

e166..  - x60 - x73 + x134 =L= 0;

e167..  - x61 + x74 - x135 =L= 0;

e168..  - x61 - x74 + x135 =L= 0;

e169..  - 11.31*i15 - 11.31*i16 - x73 + 0.5*x77 + 0.5*x80 + x134 =L= 0;

e170..  - 11.31*i15 + 11.31*i16 + x73 + 0.5*x77 + 0.5*x80 - x134 =L= 11.31;

e171..    13*i15 - 13*i16 - x74 + 0.5*x79 + 0.5*x82 + x135 =L= 13;

e172..    13*i15 + 13*i16 + x74 + 0.5*x79 + 0.5*x82 - x135 =L= 26;

e173..    x73 - x104 - x136 =L= 0;

e174..  - x73 - x104 + x136 =L= 0;

e175..    x74 - x105 - x137 =L= 0;

e176..  - x74 - x105 + x137 =L= 0;

e177..  - 11.31*i17 - 11.31*i18 - x73 + 0.5*x77 + 0.5*x84 + x136 =L= 0;

e178..  - 11.31*i17 + 11.31*i18 + x73 + 0.5*x77 + 0.5*x84 - x136 =L= 11.31;

e179..    13*i17 - 13*i18 - x74 + 0.5*x79 + 0.5*x86 + x137 =L= 13;

e180..    13*i17 + 13*i18 + x74 + 0.5*x79 + 0.5*x86 - x137 =L= 26;

e181..    x73 - x106 - x138 =L= 0;

e182..  - x73 - x106 + x138 =L= 0;

e183..    x74 - x107 - x139 =L= 0;

e184..  - x74 - x107 + x139 =L= 0;

e185..  - 11.31*i19 - 11.31*i20 - x73 + 0.5*x77 + 0.5*x88 + x138 =L= 0;

e186..  - 11.31*i19 + 11.31*i20 + x73 + 0.5*x77 + 0.5*x88 - x138 =L= 11.31;

e187..    13*i19 - 13*i20 - x74 + 0.5*x79 + 0.5*x90 + x139 =L= 13;

e188..    13*i19 + 13*i20 + x74 + 0.5*x79 + 0.5*x90 - x139 =L= 26;

e189..    x73 - x108 - x140 =L= 0;

e190..  - x73 - x108 + x140 =L= 0;

e191..    x74 - x109 - x141 =L= 0;

e192..  - x74 - x109 + x141 =L= 0;

e193..  - 11.31*i21 - 11.31*i22 - x73 + 0.5*x77 + 0.5*x92 + x140 =L= 0;

e194..  - 11.31*i21 + 11.31*i22 + x73 + 0.5*x77 + 0.5*x92 - x140 =L= 11.31;

e195..    13*i21 - 13*i22 - x74 + 0.5*x79 + 0.5*x94 + x141 =L= 13;

e196..    13*i21 + 13*i22 + x74 + 0.5*x79 + 0.5*x94 - x141 =L= 26;

e197..    x73 - x110 - x142 =L= 0;

e198..  - x73 - x110 + x142 =L= 0;

e199..    x74 - x111 - x143 =L= 0;

e200..  - x74 - x111 + x143 =L= 0;

e201..  - 11.31*i23 - 11.31*i24 - x73 + 0.5*x77 + 0.5*x96 + x142 =L= 0;

e202..  - 11.31*i23 + 11.31*i24 + x73 + 0.5*x77 + 0.5*x96 - x142 =L= 11.31;

e203..    13*i23 - 13*i24 - x74 + 0.5*x79 + 0.5*x98 + x143 =L= 13;

e204..    13*i23 + 13*i24 + x74 + 0.5*x79 + 0.5*x98 - x143 =L= 26;

e205..    x73 - x112 - x144 =L= 0;

e206..  - x73 - x112 + x144 =L= 0;

e207..    x74 - x113 - x145 =L= 0;

e208..  - x74 - x113 + x145 =L= 0;

e209..  - 11.31*i25 - 11.31*i26 - x73 + 0.5*x77 + 0.5*x100 + x144 =L= 0;

e210..  - 11.31*i25 + 11.31*i26 + x73 + 0.5*x77 + 0.5*x100 - x144 =L= 11.31;

e211..    13*i25 - 13*i26 - x74 + 0.5*x79 + 0.5*x102 + x145 =L= 13;

e212..    13*i25 + 13*i26 + x74 + 0.5*x79 + 0.5*x102 - x145 =L= 26;

e213..  - x62 + x134 - x136 =L= 0;

e214..  - x62 - x134 + x136 =L= 0;

e215..  - x63 + x135 - x137 =L= 0;

e216..  - x63 - x135 + x137 =L= 0;

e217..  - 11.31*i27 - 11.31*i28 + 0.5*x80 + 0.5*x84 - x134 + x136 =L= 0;

e218..  - 11.31*i27 + 11.31*i28 + 0.5*x80 + 0.5*x84 + x134 - x136 =L= 11.31;

e219..    13*i27 - 13*i28 + 0.5*x82 + 0.5*x86 - x135 + x137 =L= 13;

e220..    13*i27 + 13*i28 + 0.5*x82 + 0.5*x86 + x135 - x137 =L= 26;

e221..  - x114 + x134 - x138 =L= 0;

e222..  - x114 - x134 + x138 =L= 0;

e223..  - x115 + x135 - x139 =L= 0;

e224..  - x115 - x135 + x139 =L= 0;

e225..  - 11.31*i29 - 11.31*i30 + 0.5*x80 + 0.5*x88 - x134 + x138 =L= 0;

e226..  - 11.31*i29 + 11.31*i30 + 0.5*x80 + 0.5*x88 + x134 - x138 =L= 11.31;

e227..    13*i29 - 13*i30 + 0.5*x82 + 0.5*x90 - x135 + x139 =L= 13;

e228..    13*i29 + 13*i30 + 0.5*x82 + 0.5*x90 + x135 - x139 =L= 26;

e229..  - x116 + x134 - x140 =L= 0;

e230..  - x116 - x134 + x140 =L= 0;

e231..  - x117 + x135 - x141 =L= 0;

e232..  - x117 - x135 + x141 =L= 0;

e233..  - 11.31*i31 - 11.31*i32 + 0.5*x80 + 0.5*x92 - x134 + x140 =L= 0;

e234..  - 11.31*i31 + 11.31*i32 + 0.5*x80 + 0.5*x92 + x134 - x140 =L= 11.31;

e235..    13*i31 - 13*i32 + 0.5*x82 + 0.5*x94 - x135 + x141 =L= 13;

e236..    13*i31 + 13*i32 + 0.5*x82 + 0.5*x94 + x135 - x141 =L= 26;

e237..  - x118 + x134 - x142 =L= 0;

e238..  - x118 - x134 + x142 =L= 0;

e239..  - x119 + x135 - x143 =L= 0;

e240..  - x119 - x135 + x143 =L= 0;

e241..  - 11.31*i33 - 11.31*i34 + 0.5*x80 + 0.5*x96 - x134 + x142 =L= 0;

e242..  - 11.31*i33 + 11.31*i34 + 0.5*x80 + 0.5*x96 + x134 - x142 =L= 11.31;

e243..    13*i33 - 13*i34 + 0.5*x82 + 0.5*x98 - x135 + x143 =L= 13;

e244..    13*i33 + 13*i34 + 0.5*x82 + 0.5*x98 + x135 - x143 =L= 26;

e245..  - x120 + x134 - x144 =L= 0;

e246..  - x120 - x134 + x144 =L= 0;

e247..  - x121 + x135 - x145 =L= 0;

e248..  - x121 - x135 + x145 =L= 0;

e249..  - 11.31*i35 - 11.31*i36 + 0.5*x80 + 0.5*x100 - x134 + x144 =L= 0;

e250..  - 11.31*i35 + 11.31*i36 + 0.5*x80 + 0.5*x100 + x134 - x144 =L= 11.31;

e251..    13*i35 - 13*i36 + 0.5*x82 + 0.5*x102 - x135 + x145 =L= 13;

e252..    13*i35 + 13*i36 + 0.5*x82 + 0.5*x102 + x135 - x145 =L= 26;

e253..  - x64 + x136 - x138 =L= 0;

e254..  - x64 - x136 + x138 =L= 0;

e255..  - x65 + x137 - x139 =L= 0;

e256..  - x65 - x137 + x139 =L= 0;

e257..  - 11.31*i37 - 11.31*i38 + 0.5*x84 + 0.5*x88 - x136 + x138 =L= 0;

e258..  - 11.31*i37 + 11.31*i38 + 0.5*x84 + 0.5*x88 + x136 - x138 =L= 11.31;

e259..    13*i37 - 13*i38 + 0.5*x86 + 0.5*x90 - x137 + x139 =L= 13;

e260..    13*i37 + 13*i38 + 0.5*x86 + 0.5*x90 + x137 - x139 =L= 26;

e261..  - x122 + x136 - x140 =L= 0;

e262..  - x122 - x136 + x140 =L= 0;

e263..  - x123 + x137 - x141 =L= 0;

e264..  - x123 - x137 + x141 =L= 0;

e265..  - 11.31*i39 - 11.31*i40 + 0.5*x84 + 0.5*x92 - x136 + x140 =L= 0;

e266..  - 11.31*i39 + 11.31*i40 + 0.5*x84 + 0.5*x92 + x136 - x140 =L= 11.31;

e267..    13*i39 - 13*i40 + 0.5*x86 + 0.5*x94 - x137 + x141 =L= 13;

e268..    13*i39 + 13*i40 + 0.5*x86 + 0.5*x94 + x137 - x141 =L= 26;

e269..  - x124 + x136 - x142 =L= 0;

e270..  - x124 - x136 + x142 =L= 0;

e271..  - x125 + x137 - x143 =L= 0;

e272..  - x125 - x137 + x143 =L= 0;

e273..  - 11.31*i41 - 11.31*i42 + 0.5*x84 + 0.5*x96 - x136 + x142 =L= 0;

e274..  - 11.31*i41 + 11.31*i42 + 0.5*x84 + 0.5*x96 + x136 - x142 =L= 11.31;

e275..    13*i41 - 13*i42 + 0.5*x86 + 0.5*x98 - x137 + x143 =L= 13;

e276..    13*i41 + 13*i42 + 0.5*x86 + 0.5*x98 + x137 - x143 =L= 26;

e277..  - x126 + x136 - x144 =L= 0;

e278..  - x126 - x136 + x144 =L= 0;

e279..  - x127 + x137 - x145 =L= 0;

e280..  - x127 - x137 + x145 =L= 0;

e281..  - 11.31*i43 - 11.31*i44 + 0.5*x84 + 0.5*x100 - x136 + x144 =L= 0;

e282..  - 11.31*i43 + 11.31*i44 + 0.5*x84 + 0.5*x100 + x136 - x144 =L= 11.31;

e283..    13*i43 - 13*i44 + 0.5*x86 + 0.5*x102 - x137 + x145 =L= 13;

e284..    13*i43 + 13*i44 + 0.5*x86 + 0.5*x102 + x137 - x145 =L= 26;

e285..  - x66 + x138 - x140 =L= 0;

e286..  - x66 - x138 + x140 =L= 0;

e287..  - x67 + x139 - x141 =L= 0;

e288..  - x67 - x139 + x141 =L= 0;

e289..  - 11.31*i45 - 11.31*i46 + 0.5*x88 + 0.5*x92 - x138 + x140 =L= 0;

e290..  - 11.31*i45 + 11.31*i46 + 0.5*x88 + 0.5*x92 + x138 - x140 =L= 11.31;

e291..    13*i45 - 13*i46 + 0.5*x90 + 0.5*x94 - x139 + x141 =L= 13;

e292..    13*i45 + 13*i46 + 0.5*x90 + 0.5*x94 + x139 - x141 =L= 26;

e293..  - x128 + x138 - x142 =L= 0;

e294..  - x128 - x138 + x142 =L= 0;

e295..  - x129 + x139 - x143 =L= 0;

e296..  - x129 - x139 + x143 =L= 0;

e297..  - 11.31*i47 - 11.31*i48 + 0.5*x88 + 0.5*x96 - x138 + x142 =L= 0;

e298..  - 11.31*i47 + 11.31*i48 + 0.5*x88 + 0.5*x96 + x138 - x142 =L= 11.31;

e299..    13*i47 - 13*i48 + 0.5*x90 + 0.5*x98 - x139 + x143 =L= 13;

e300..    13*i47 + 13*i48 + 0.5*x90 + 0.5*x98 + x139 - x143 =L= 26;

e301..  - x130 + x138 - x144 =L= 0;

e302..  - x130 - x138 + x144 =L= 0;

e303..  - x131 + x139 - x145 =L= 0;

e304..  - x131 - x139 + x145 =L= 0;

e305..  - 11.31*i49 - 11.31*i50 + 0.5*x88 + 0.5*x100 - x138 + x144 =L= 0;

e306..  - 11.31*i49 + 11.31*i50 + 0.5*x88 + 0.5*x100 + x138 - x144 =L= 11.31;

e307..    13*i49 - 13*i50 + 0.5*x90 + 0.5*x102 - x139 + x145 =L= 13;

e308..    13*i49 + 13*i50 + 0.5*x90 + 0.5*x102 + x139 - x145 =L= 26;

e309..  - x68 + x140 - x142 =L= 0;

e310..  - x68 - x140 + x142 =L= 0;

e311..  - x69 + x141 - x143 =L= 0;

e312..  - x69 - x141 + x143 =L= 0;

e313..  - 11.31*i51 - 11.31*i52 + 0.5*x92 + 0.5*x96 - x140 + x142 =L= 0;

e314..  - 11.31*i51 + 11.31*i52 + 0.5*x92 + 0.5*x96 + x140 - x142 =L= 11.31;

e315..    13*i51 - 13*i52 + 0.5*x94 + 0.5*x98 - x141 + x143 =L= 13;

e316..    13*i51 + 13*i52 + 0.5*x94 + 0.5*x98 + x141 - x143 =L= 26;

e317..  - x132 + x140 - x144 =L= 0;

e318..  - x132 - x140 + x144 =L= 0;

e319..  - x133 + x141 - x145 =L= 0;

e320..  - x133 - x141 + x145 =L= 0;

e321..  - 11.31*i53 - 11.31*i54 + 0.5*x92 + 0.5*x100 - x140 + x144 =L= 0;

e322..  - 11.31*i53 + 11.31*i54 + 0.5*x92 + 0.5*x100 + x140 - x144 =L= 11.31;

e323..    13*i53 - 13*i54 + 0.5*x94 + 0.5*x102 - x141 + x145 =L= 13;

e324..    13*i53 + 13*i54 + 0.5*x94 + 0.5*x102 + x141 - x145 =L= 26;

e325..  - x70 + x142 - x144 =L= 0;

e326..  - x70 - x142 + x144 =L= 0;

e327..  - x71 + x143 - x145 =L= 0;

e328..  - x71 - x143 + x145 =L= 0;

e329..  - 11.31*i55 - 11.31*i56 + 0.5*x96 + 0.5*x100 - x142 + x144 =L= 0;

e330..  - 11.31*i55 + 11.31*i56 + 0.5*x96 + 0.5*x100 + x142 - x144 =L= 11.31;

e331..    13*i55 - 13*i56 + 0.5*x98 + 0.5*x102 - x143 + x145 =L= 13;

e332..    13*i55 + 13*i56 + 0.5*x98 + 0.5*x102 + x143 - x145 =L= 26;

e333.. 16/x76 - x78 =L= 0;

e334.. 16/x78 - x76 =L= 0;

e335.. 16/x77 - x79 =L= 0;

e336.. 16/x79 - x77 =L= 0;

e337.. 16/x80 - x82 =L= 0;

e338.. 16/x82 - x80 =L= 0;

e339.. 36/x84 - x86 =L= 0;

e340.. 36/x86 - x84 =L= 0;

e341.. 36/x88 - x90 =L= 0;

e342.. 36/x90 - x88 =L= 0;

e343.. 9/x92 - x94 =L= 0;

e344.. 9/x94 - x92 =L= 0;

e345.. 9/x96 - x98 =L= 0;

e346.. 9/x98 - x96 =L= 0;

e347.. 9/x100 - x102 =L= 0;

e348.. 9/x102 - x100 =L= 0;

* set non default bounds

i1.up = 100; 
i2.up = 100; 
i3.up = 100; 
i4.up = 100; 
i5.up = 100; 
i6.up = 100; 
i7.up = 100; 
i8.up = 100; 
i9.up = 100; 
i10.up = 100; 
i11.up = 100; 
i12.up = 100; 
i13.up = 100; 
i14.up = 100; 
i15.up = 100; 
i16.up = 100; 
i17.up = 100; 
i18.up = 100; 
i19.up = 100; 
i20.up = 100; 
i21.up = 100; 
i22.up = 100; 
i23.up = 100; 
i24.up = 100; 
i25.up = 100; 
i26.up = 100; 
i27.up = 100; 
i28.up = 100; 
i29.up = 100; 
i30.up = 100; 
i31.up = 100; 
i32.up = 100; 
i33.up = 100; 
i34.up = 100; 
i35.up = 100; 
i36.up = 100; 
i37.up = 100; 
i38.up = 100; 
i39.up = 100; 
i40.up = 100; 
i41.up = 100; 
i42.up = 100; 
i43.up = 100; 
i44.up = 100; 
i45.up = 100; 
i46.up = 100; 
i47.up = 100; 
i48.up = 100; 
i49.up = 100; 
i50.up = 100; 
i51.up = 100; 
i52.up = 100; 
i53.up = 100; 
i54.up = 100; 
i55.up = 100; 
i56.up = 100; 
x76.lo = 2.8284; x76.up = 5.6569; 
x77.lo = 2.8284; x77.up = 5.6569; 
x78.lo = 2.8284; x78.up = 5.6569; 
x79.lo = 2.8284; x79.up = 5.6569; 
x80.lo = 2.8284; x80.up = 5.6569; 
x82.lo = 2.8284; x82.up = 5.6569; 
x84.lo = 4.2426; x84.up = 8.4853; 
x86.lo = 4.2426; x86.up = 8.4853; 
x88.lo = 4.2426; x88.up = 8.4853; 
x90.lo = 4.2426; x90.up = 8.4853; 
x92.lo = 2.1213; x92.up = 4.2426; 
x94.lo = 2.1213; x94.up = 4.2426; 
x96.lo = 2.1213; x96.up = 4.2426; 
x98.lo = 2.1213; x98.up = 4.2426; 
x100.lo = 2.1213; x100.up = 4.2426; 
x102.lo = 2.1213; x102.up = 4.2426; 

$if set nostart $goto modeldef
* set non default levels


* set non default marginals


$label modeldef
Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set MINLP $set MINLP MINLP
Solve m using %MINLP% minimizing objvar;
