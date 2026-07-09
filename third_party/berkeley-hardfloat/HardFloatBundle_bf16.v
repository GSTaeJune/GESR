`timescale 1ns/1ps
// Berkeley HardFloat - bf16 (e8/s8) subset. Generated via Chisel 3.5.6 (sbt,
// Java 17 with JAVA_TOOL_OPTIONS=--add-opens=...). See VENDORING.md.
// Modules: INToRecFN_i32_e8_s8 (+ helper RoundAnyRawFNToRecFN_ie6_is32_oe8_os8),
//          FNFromRecFN_bf16_wrapper. NO bf16 AddRecFN (sigWidth=8 unsupported by
//          HardFloat's adder core) - bf16 add is done in the fp32 domain.

module RoundAnyRawFNToRecFN_ie6_is32_oe8_os8(
  input         io_in_isZero,
  input         io_in_sign,
  input  [7:0]  io_in_sExp,
  input  [32:0] io_in_sig,
  input  [2:0]  io_roundingMode,
  output [16:0] io_out,
  output [4:0]  io_exceptionFlags
);
  wire  roundingMode_near_even = io_roundingMode == 3'h0; // @[RoundAnyRawFNToRecFN.scala 90:53]
  wire  roundingMode_min = io_roundingMode == 3'h2; // @[RoundAnyRawFNToRecFN.scala 92:53]
  wire  roundingMode_max = io_roundingMode == 3'h3; // @[RoundAnyRawFNToRecFN.scala 93:53]
  wire  roundingMode_near_maxMag = io_roundingMode == 3'h4; // @[RoundAnyRawFNToRecFN.scala 94:53]
  wire  roundingMode_odd = io_roundingMode == 3'h6; // @[RoundAnyRawFNToRecFN.scala 95:53]
  wire  roundMagUp = roundingMode_min & io_in_sign | roundingMode_max & ~io_in_sign; // @[RoundAnyRawFNToRecFN.scala 98:42]
  wire [8:0] _GEN_0 = {{1{io_in_sExp[7]}},io_in_sExp}; // @[RoundAnyRawFNToRecFN.scala 104:25]
  wire [9:0] _sAdjustedExp_T = $signed(_GEN_0) + 9'shc0; // @[RoundAnyRawFNToRecFN.scala 104:25]
  wire [9:0] sAdjustedExp = {1'b0,$signed(_sAdjustedExp_T[8:0])}; // @[RoundAnyRawFNToRecFN.scala 106:31]
  wire  _adjustedSig_T_2 = |io_in_sig[22:0]; // @[RoundAnyRawFNToRecFN.scala 117:60]
  wire [10:0] adjustedSig = {io_in_sig[32:23],_adjustedSig_T_2}; // @[RoundAnyRawFNToRecFN.scala 116:66]
  wire [10:0] _roundPosBit_T = adjustedSig & 11'h2; // @[RoundAnyRawFNToRecFN.scala 164:40]
  wire  roundPosBit = |_roundPosBit_T; // @[RoundAnyRawFNToRecFN.scala 164:56]
  wire [10:0] _anyRoundExtra_T = adjustedSig & 11'h1; // @[RoundAnyRawFNToRecFN.scala 165:42]
  wire  anyRoundExtra = |_anyRoundExtra_T; // @[RoundAnyRawFNToRecFN.scala 165:62]
  wire  anyRound = roundPosBit | anyRoundExtra; // @[RoundAnyRawFNToRecFN.scala 166:36]
  wire  _roundIncr_T_1 = (roundingMode_near_even | roundingMode_near_maxMag) & roundPosBit; // @[RoundAnyRawFNToRecFN.scala 169:67]
  wire  _roundIncr_T_2 = roundMagUp & anyRound; // @[RoundAnyRawFNToRecFN.scala 171:29]
  wire  roundIncr = _roundIncr_T_1 | _roundIncr_T_2; // @[RoundAnyRawFNToRecFN.scala 170:31]
  wire [10:0] _roundedSig_T = adjustedSig | 11'h3; // @[RoundAnyRawFNToRecFN.scala 174:32]
  wire [9:0] _roundedSig_T_2 = _roundedSig_T[10:2] + 9'h1; // @[RoundAnyRawFNToRecFN.scala 174:49]
  wire  _roundedSig_T_4 = ~anyRoundExtra; // @[RoundAnyRawFNToRecFN.scala 176:30]
  wire [9:0] _roundedSig_T_7 = roundingMode_near_even & roundPosBit & _roundedSig_T_4 ? 10'h1 : 10'h0; // @[RoundAnyRawFNToRecFN.scala 175:25]
  wire [9:0] _roundedSig_T_8 = ~_roundedSig_T_7; // @[RoundAnyRawFNToRecFN.scala 175:21]
  wire [9:0] _roundedSig_T_9 = _roundedSig_T_2 & _roundedSig_T_8; // @[RoundAnyRawFNToRecFN.scala 174:57]
  wire [10:0] _roundedSig_T_11 = adjustedSig & 11'h7fc; // @[RoundAnyRawFNToRecFN.scala 180:30]
  wire [9:0] _roundedSig_T_15 = roundingMode_odd & anyRound ? 10'h1 : 10'h0; // @[RoundAnyRawFNToRecFN.scala 181:24]
  wire [9:0] _GEN_1 = {{1'd0}, _roundedSig_T_11[10:2]}; // @[RoundAnyRawFNToRecFN.scala 180:47]
  wire [9:0] _roundedSig_T_16 = _GEN_1 | _roundedSig_T_15; // @[RoundAnyRawFNToRecFN.scala 180:47]
  wire [9:0] roundedSig = roundIncr ? _roundedSig_T_9 : _roundedSig_T_16; // @[RoundAnyRawFNToRecFN.scala 173:16]
  wire [2:0] _sRoundedExp_T_1 = {1'b0,$signed(roundedSig[9:8])}; // @[RoundAnyRawFNToRecFN.scala 185:76]
  wire [9:0] _GEN_2 = {{7{_sRoundedExp_T_1[2]}},_sRoundedExp_T_1}; // @[RoundAnyRawFNToRecFN.scala 185:40]
  wire [10:0] sRoundedExp = $signed(sAdjustedExp) + $signed(_GEN_2); // @[RoundAnyRawFNToRecFN.scala 185:40]
  wire [8:0] common_expOut = sRoundedExp[8:0]; // @[RoundAnyRawFNToRecFN.scala 187:37]
  wire [6:0] common_fractOut = roundedSig[6:0]; // @[RoundAnyRawFNToRecFN.scala 191:27]
  wire  commonCase = ~io_in_isZero; // @[RoundAnyRawFNToRecFN.scala 237:64]
  wire  inexact = commonCase & anyRound; // @[RoundAnyRawFNToRecFN.scala 240:43]
  wire [8:0] _expOut_T_1 = io_in_isZero ? 9'h1c0 : 9'h0; // @[RoundAnyRawFNToRecFN.scala 253:18]
  wire [8:0] _expOut_T_2 = ~_expOut_T_1; // @[RoundAnyRawFNToRecFN.scala 253:14]
  wire [8:0] expOut = common_expOut & _expOut_T_2; // @[RoundAnyRawFNToRecFN.scala 252:24]
  wire [6:0] fractOut = io_in_isZero ? 7'h0 : common_fractOut; // @[RoundAnyRawFNToRecFN.scala 280:12]
  wire [9:0] _io_out_T = {io_in_sign,expOut}; // @[RoundAnyRawFNToRecFN.scala 286:23]
  assign io_out = {_io_out_T,fractOut}; // @[RoundAnyRawFNToRecFN.scala 286:33]
  assign io_exceptionFlags = {4'h0,inexact}; // @[RoundAnyRawFNToRecFN.scala 288:66]
endmodule
module INToRecFN_i32_e8_s8(
  input         io_signedIn,
  input  [31:0] io_in,
  input  [2:0]  io_roundingMode,
  input         io_detectTininess,
  output [16:0] io_out,
  output [4:0]  io_exceptionFlags
);
  wire  roundAnyRawFNToRecFN_io_in_isZero; // @[INToRecFN.scala 60:15]
  wire  roundAnyRawFNToRecFN_io_in_sign; // @[INToRecFN.scala 60:15]
  wire [7:0] roundAnyRawFNToRecFN_io_in_sExp; // @[INToRecFN.scala 60:15]
  wire [32:0] roundAnyRawFNToRecFN_io_in_sig; // @[INToRecFN.scala 60:15]
  wire [2:0] roundAnyRawFNToRecFN_io_roundingMode; // @[INToRecFN.scala 60:15]
  wire [16:0] roundAnyRawFNToRecFN_io_out; // @[INToRecFN.scala 60:15]
  wire [4:0] roundAnyRawFNToRecFN_io_exceptionFlags; // @[INToRecFN.scala 60:15]
  wire  intAsRawFloat_sign = io_signedIn & io_in[31]; // @[rawFloatFromIN.scala 51:29]
  wire [31:0] _intAsRawFloat_absIn_T_1 = 32'h0 - io_in; // @[rawFloatFromIN.scala 52:31]
  wire [31:0] intAsRawFloat_absIn = intAsRawFloat_sign ? _intAsRawFloat_absIn_T_1 : io_in; // @[rawFloatFromIN.scala 52:24]
  wire [63:0] _intAsRawFloat_extAbsIn_T = {32'h0,intAsRawFloat_absIn}; // @[rawFloatFromIN.scala 53:44]
  wire [31:0] intAsRawFloat_extAbsIn = _intAsRawFloat_extAbsIn_T[31:0]; // @[rawFloatFromIN.scala 53:53]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_32 = intAsRawFloat_extAbsIn[1] ? 5'h1e : 5'h1f; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_33 = intAsRawFloat_extAbsIn[2] ? 5'h1d :
    _intAsRawFloat_adjustedNormDist_T_32; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_34 = intAsRawFloat_extAbsIn[3] ? 5'h1c :
    _intAsRawFloat_adjustedNormDist_T_33; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_35 = intAsRawFloat_extAbsIn[4] ? 5'h1b :
    _intAsRawFloat_adjustedNormDist_T_34; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_36 = intAsRawFloat_extAbsIn[5] ? 5'h1a :
    _intAsRawFloat_adjustedNormDist_T_35; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_37 = intAsRawFloat_extAbsIn[6] ? 5'h19 :
    _intAsRawFloat_adjustedNormDist_T_36; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_38 = intAsRawFloat_extAbsIn[7] ? 5'h18 :
    _intAsRawFloat_adjustedNormDist_T_37; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_39 = intAsRawFloat_extAbsIn[8] ? 5'h17 :
    _intAsRawFloat_adjustedNormDist_T_38; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_40 = intAsRawFloat_extAbsIn[9] ? 5'h16 :
    _intAsRawFloat_adjustedNormDist_T_39; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_41 = intAsRawFloat_extAbsIn[10] ? 5'h15 :
    _intAsRawFloat_adjustedNormDist_T_40; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_42 = intAsRawFloat_extAbsIn[11] ? 5'h14 :
    _intAsRawFloat_adjustedNormDist_T_41; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_43 = intAsRawFloat_extAbsIn[12] ? 5'h13 :
    _intAsRawFloat_adjustedNormDist_T_42; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_44 = intAsRawFloat_extAbsIn[13] ? 5'h12 :
    _intAsRawFloat_adjustedNormDist_T_43; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_45 = intAsRawFloat_extAbsIn[14] ? 5'h11 :
    _intAsRawFloat_adjustedNormDist_T_44; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_46 = intAsRawFloat_extAbsIn[15] ? 5'h10 :
    _intAsRawFloat_adjustedNormDist_T_45; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_47 = intAsRawFloat_extAbsIn[16] ? 5'hf :
    _intAsRawFloat_adjustedNormDist_T_46; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_48 = intAsRawFloat_extAbsIn[17] ? 5'he :
    _intAsRawFloat_adjustedNormDist_T_47; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_49 = intAsRawFloat_extAbsIn[18] ? 5'hd :
    _intAsRawFloat_adjustedNormDist_T_48; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_50 = intAsRawFloat_extAbsIn[19] ? 5'hc :
    _intAsRawFloat_adjustedNormDist_T_49; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_51 = intAsRawFloat_extAbsIn[20] ? 5'hb :
    _intAsRawFloat_adjustedNormDist_T_50; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_52 = intAsRawFloat_extAbsIn[21] ? 5'ha :
    _intAsRawFloat_adjustedNormDist_T_51; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_53 = intAsRawFloat_extAbsIn[22] ? 5'h9 :
    _intAsRawFloat_adjustedNormDist_T_52; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_54 = intAsRawFloat_extAbsIn[23] ? 5'h8 :
    _intAsRawFloat_adjustedNormDist_T_53; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_55 = intAsRawFloat_extAbsIn[24] ? 5'h7 :
    _intAsRawFloat_adjustedNormDist_T_54; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_56 = intAsRawFloat_extAbsIn[25] ? 5'h6 :
    _intAsRawFloat_adjustedNormDist_T_55; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_57 = intAsRawFloat_extAbsIn[26] ? 5'h5 :
    _intAsRawFloat_adjustedNormDist_T_56; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_58 = intAsRawFloat_extAbsIn[27] ? 5'h4 :
    _intAsRawFloat_adjustedNormDist_T_57; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_59 = intAsRawFloat_extAbsIn[28] ? 5'h3 :
    _intAsRawFloat_adjustedNormDist_T_58; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_60 = intAsRawFloat_extAbsIn[29] ? 5'h2 :
    _intAsRawFloat_adjustedNormDist_T_59; // @[Mux.scala 47:70]
  wire [4:0] _intAsRawFloat_adjustedNormDist_T_61 = intAsRawFloat_extAbsIn[30] ? 5'h1 :
    _intAsRawFloat_adjustedNormDist_T_60; // @[Mux.scala 47:70]
  wire [4:0] intAsRawFloat_adjustedNormDist = intAsRawFloat_extAbsIn[31] ? 5'h0 : _intAsRawFloat_adjustedNormDist_T_61; // @[Mux.scala 47:70]
  wire [62:0] _GEN_0 = {{31'd0}, intAsRawFloat_extAbsIn}; // @[rawFloatFromIN.scala 56:22]
  wire [62:0] _intAsRawFloat_sig_T = _GEN_0 << intAsRawFloat_adjustedNormDist; // @[rawFloatFromIN.scala 56:22]
  wire [31:0] intAsRawFloat_sig = _intAsRawFloat_sig_T[31:0]; // @[rawFloatFromIN.scala 56:41]
  wire [4:0] _intAsRawFloat_out_sExp_T_1 = ~intAsRawFloat_adjustedNormDist; // @[rawFloatFromIN.scala 64:36]
  wire [6:0] _intAsRawFloat_out_sExp_T_2 = {2'h2,_intAsRawFloat_out_sExp_T_1}; // @[rawFloatFromIN.scala 64:33]
  RoundAnyRawFNToRecFN_ie6_is32_oe8_os8 roundAnyRawFNToRecFN ( // @[INToRecFN.scala 60:15]
    .io_in_isZero(roundAnyRawFNToRecFN_io_in_isZero),
    .io_in_sign(roundAnyRawFNToRecFN_io_in_sign),
    .io_in_sExp(roundAnyRawFNToRecFN_io_in_sExp),
    .io_in_sig(roundAnyRawFNToRecFN_io_in_sig),
    .io_roundingMode(roundAnyRawFNToRecFN_io_roundingMode),
    .io_out(roundAnyRawFNToRecFN_io_out),
    .io_exceptionFlags(roundAnyRawFNToRecFN_io_exceptionFlags)
  );
  assign io_out = roundAnyRawFNToRecFN_io_out; // @[INToRecFN.scala 73:23]
  assign io_exceptionFlags = roundAnyRawFNToRecFN_io_exceptionFlags; // @[INToRecFN.scala 74:23]
  assign roundAnyRawFNToRecFN_io_in_isZero = ~intAsRawFloat_sig[31]; // @[rawFloatFromIN.scala 62:23]
  assign roundAnyRawFNToRecFN_io_in_sign = io_signedIn & io_in[31]; // @[rawFloatFromIN.scala 51:29]
  assign roundAnyRawFNToRecFN_io_in_sExp = {1'b0,$signed(_intAsRawFloat_out_sExp_T_2)}; // @[rawFloatFromIN.scala 64:72]
  assign roundAnyRawFNToRecFN_io_in_sig = {{1'd0}, intAsRawFloat_sig}; // @[rawFloatFromIN.scala 59:23 65:20]
  assign roundAnyRawFNToRecFN_io_roundingMode = io_roundingMode; // @[INToRecFN.scala 71:44]
endmodule

module FNFromRecFN_bf16_wrapper(
  input  [16:0] in,
  output [15:0] out
);
  wire [8:0] out_rawIn_exp = in[15:7]; // @[rawFloatFromRecFN.scala 51:21]
  wire  out_rawIn_isZero = out_rawIn_exp[8:6] == 3'h0; // @[rawFloatFromRecFN.scala 52:53]
  wire  out_rawIn_isSpecial = out_rawIn_exp[8:7] == 2'h3; // @[rawFloatFromRecFN.scala 53:53]
  wire  out_rawIn__isNaN = out_rawIn_isSpecial & out_rawIn_exp[6]; // @[rawFloatFromRecFN.scala 56:33]
  wire  out_rawIn__isInf = out_rawIn_isSpecial & ~out_rawIn_exp[6]; // @[rawFloatFromRecFN.scala 57:33]
  wire  out_rawIn__sign = in[16]; // @[rawFloatFromRecFN.scala 59:25]
  wire [9:0] out_rawIn__sExp = {1'b0,$signed(out_rawIn_exp)}; // @[rawFloatFromRecFN.scala 60:27]
  wire  _out_rawIn_out_sig_T = ~out_rawIn_isZero; // @[rawFloatFromRecFN.scala 61:35]
  wire [8:0] out_rawIn__sig = {1'h0,_out_rawIn_out_sig_T,in[6:0]}; // @[rawFloatFromRecFN.scala 61:44]
  wire  out_isSubnormal = $signed(out_rawIn__sExp) < 10'sh82; // @[fNFromRecFN.scala 51:38]
  wire [2:0] out_denormShiftDist = 3'h1 - out_rawIn__sExp[2:0]; // @[fNFromRecFN.scala 52:35]
  wire [7:0] _out_denormFract_T_1 = out_rawIn__sig[8:1] >> out_denormShiftDist; // @[fNFromRecFN.scala 53:42]
  wire [6:0] out_denormFract = _out_denormFract_T_1[6:0]; // @[fNFromRecFN.scala 53:60]
  wire [7:0] _out_expOut_T_2 = out_rawIn__sExp[7:0] - 8'h81; // @[fNFromRecFN.scala 58:45]
  wire [7:0] _out_expOut_T_3 = out_isSubnormal ? 8'h0 : _out_expOut_T_2; // @[fNFromRecFN.scala 56:16]
  wire  _out_expOut_T_4 = out_rawIn__isNaN | out_rawIn__isInf; // @[fNFromRecFN.scala 60:44]
  wire [7:0] _out_expOut_T_6 = _out_expOut_T_4 ? 8'hff : 8'h0; // @[Bitwise.scala 77:12]
  wire [7:0] out_expOut = _out_expOut_T_3 | _out_expOut_T_6; // @[fNFromRecFN.scala 60:15]
  wire [6:0] _out_fractOut_T_1 = out_rawIn__isInf ? 7'h0 : out_rawIn__sig[6:0]; // @[fNFromRecFN.scala 64:20]
  wire [6:0] out_fractOut = out_isSubnormal ? out_denormFract : _out_fractOut_T_1; // @[fNFromRecFN.scala 62:16]
  wire [8:0] out_hi = {out_rawIn__sign,out_expOut}; // @[Cat.scala 33:92]
  assign out = {out_hi,out_fractOut}; // @[Cat.scala 33:92]
endmodule
