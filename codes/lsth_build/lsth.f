c  routine for evaluation of lsth h3 surface
c  this version uses optimized h2 spline routine called splin2.f
c to compile on suns: f77 -c -Nl55 lsth2.f
c sample call:
c     call vlsth(X,E,E1,E2,E3,ideriv,ipr,isurf)
C
      subroutine splid2(N,X,F,W,IJ,Y,TAB)
c-----------------------------------------------------------------
c  optimized version of splid2 subroutine ... oct 20/90
c  super efficient search method oct20/90
C    THIS MODULE CALCULATES ENERGY AND DERIVATIVES FOR A GIVEN
C    DISTANCE y (used only for y<10 bohrs)
C    TAB(1)=ENERGY, TAB(2)=1ST DERIVATIVE, TAB(3)=2ND DERIVATIVE
c  method used here makes use of the equal spacing between some
c  of the spline points
c-----------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION X(n),F(n),W(n),TAB(3)
c    if y < 1st spline point, set i=1 and go on to 35
c    if y > 1st spline point, search for spline point larger than y:
      if(y.le.0.4d0)then
	i=1
	go to 35
      end if
c    if y > last spl.pt., set i=n-1 and go on to 35
c    if y < last spl.pt., search thru spl.pts
      if(y.ge.10.d0)then
	 i=n-1
	 go to 35
      end if
c   split spline points into 6 regions and only search the
c   appropriate region
      if( y .gt.  7.d0 ) then
	if(y.le.8.d0)then
	  i = (y-7.d0)/0.2d0 + 77
	  go to 35
	else
	  i = 82
	  go to 30
	end if
      else
	if(y.ge.1.5d0)then
	   i = (y-1.5d0)/0.1d0 + 22
	   go to 35
	else
	  if(y.ge.1.35d0)then
	    i = 15
	    go to 30
	  end if
	  if(y.le.0.8d0)then
	    i = (y-0.4d0)/0.05d0 + 1
	    go to 35
	  else
	    i = (y-0.8d0)/0.1d0 + 9
	    go to 35
	  end if
	end if
      end if
c    search through a few points for the right one
 0030 continue
      DO K= i, N
         IF( X(K) .GT. Y ) then
	    i = k - 1
	    GOTO 35
	 end if
      end do
c--------
c now that we know the appropriate spline segment, interpolate and
c find the derivatives
   35 MI=(I-1)*IJ+1
      KI=MI+IJ
      xati=x(i)
      xip1=x(i+1)
      wmi = w(mi)
      wki = w(ki)
      fki = f(ki)
      fmi = f(mi)
      FLK=xip1  -xati
      ta =xip1  -Y
      ta2=ta*ta
      ta3=ta*ta2
      tb =Y-xati
      tb2=tb*tb
      tb3=tb*tb2
      A=( wmi   * ta3           + wki   * tb3         ) /(6.*FLK)
      B=(fki  /FLK-wki  *FLK/6.)*  tb
      C=( fmi /FLK-FLK*wmi  /6.)*  ta
      TAB(1)=A+B+C
      A=( wki   * tb2         - wmi  * ta2           )/(2.*FLK)
      B=(fki  - fmi )/FLK
      C=FLK*(wmi  -wki  )/6.
      TAB(2)=A+B+C
      TAB(3)=( wmi  * ta         + wki   * tb       )/FLK
      RETURN
      END
c
      subroutine HHPOT(x,s)
CCC********************************************************* module 11 **
c  changed from a function to a subroutine on may 9/90
C  CALCULATES H-H POTENTIAL
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION S(3)
      COMMON/POTCOM/C6,C8,RKW(87),EKW(87),WKW(87)
C    CONVERT INTERNUCLEAR SEPARATION TO BOHRS
      IF(x.GT.10.)CALL VBIGR(X,S)
      IF(X.LE.10.)CALL SPLID2(87,RKW,EKW,WKW,1,X,S)
      RETURN
      END
C
      subroutine vlsth(X,E,E1,E2,E3,ideriv,ipr,isurf)
c------------------------------------------------------- module 20 ----c
c apr13/95 ... subr. name changed from v to vlsth by wjk
C CALCULATE THE POTENTIAL (CALLED FROM SUBR.PIP)
C    TO CALCULATE ONLY THE ENERGY SET ideriv to 0
C    TO CALCULATE DERIVATIVES AS WELL, SET IT TO 1 or greater
C    NOTE THIS IS NOT AN OPTIMIZED CODE FROM THE POINT OF VIEW OF
C    COMPACTNESS OF THE CODING
c  if isurf=0 ... use the spline h2 potential
c  if isurf=1 ... use Schwenke's h2 potential
C  X == array of 3 distances
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/VCOM/C,A,A1,F,FNS,F1,F2,F3,AN1,AN2,AN3,AN4,B1,B2,B3,
     .    W1,W2,W3,D1,D2,D3,D4,XL1,XL2
      DIMENSION X(3),S1(3),S2(3),S3(3)
C  calculate London energy:
      EF1=EXP(F*X(1))
      EF2=EXP(F*X(2))
      EF3=EXP(F*X(3))
c     r1**2 , r2**2, r3**2
      X21=X(1)*X(1)
      X22=X(2)*X(2)
      X23=X(3)*X(3)
c     triplet energy for r1     see eq.(10)
      T1=C*(A+X(1)+A1*X21)/EF1
      T2=C*(A+X(2)+A1*X22)/EF2
      T3=C*(A+X(3)+A1*X23)/EF3
      CALL VH2(X,S1,S2,S3,isurf)
c     {singlet energy and derivatives}
c     see eq.(7)
      XQ1=S1(1)+T1
      XQ2=S2(1)+T2
      XQ3=S3(1)+T3
c     see eq.(8)
      XJ1=S1(1)-T1
      XJ2=S2(1)-T2
      XJ3=S3(1)-T3
c     sum of Q values (see eq.(6))
      XQ=(XQ1+XQ2+XQ3)/2.
      XJ=SQRT(((XJ1-XJ2)**2+(XJ2-XJ3)**2+(XJ3-XJ1)**2)/8.)
      ELOND=XQ-XJ
C    ENS
      WNT=(X(1)-X(2))*(X(2)-X(3))*(X(3)-X(1))
      WN=ABS(WNT)
      WN2=WN*WN
      WN3=WN2*WN
      WN4=WN3*WN
      WN5=WN4*WN
      R=X(1)+X(2)+X(3)
      R2=R*R
      R3=R2*R
      EXNS=EXP(FNS*R3)
      ENS=(AN1*WN2+AN2*WN3+AN3*WN4+AN4*WN5)/EXNS
C  NONLINEAR CORRECTIONS
C
      COS=(X21+X22+X23)/2.
      COS1=(X21-COS)/X(2)/X(3)
      COS2=(X22-COS)/X(1)/X(3)
      COS3=(X23-COS)/X(1)/X(2)
c     wb = b1
      WB=COS1+COS2+COS3+1.
      WB2=WB*WB
      WB3=WB2*WB
      WB4=WB3*WB
      EXF1=EXP(F1*R)
      EXF2=EXP(F2*R2)
      EXF3=EXP(F3*R)
      EB1T=(B1+B2*R)/EXF1
      EB3T=(XL1+XL2*R2)/EXF3
c     eb1 = vb1 + vb5
      EB1=WB*(EB1T+EB3T)
      EB2=(WB2*W1+WB3*W2+WB4*W3)/EXF2
c     {EB2=Vb2
      EQ=(X(1)-X(2))**2+(X(2)-X(3))**2+(X(3)-X(1))**2
c     {EQ=B3
      RI=1./X(1)+1./X(2)+1./X(3)
c     {RI=B2
      EB4A=WB*D1/EXF1+WB2*D2/EXF2
      EB4B=D3/EXF1+D4/EXF2
      EB4=EB4A*RI+EB4B*WB*EQ
c     {EB4=Vb3+Vb4
      E=ELOND+ENS+EB1+EB2+EB4
      if(ipr.gt.0)then
         write(6,*) ' --------- output from subr.v ---------'
         write(6,6100) (x(i),i=1,3)
         write(6,6050) wb,ri,eq
         write(6,6000) elond,ens,eb1,eb2,eb4
 6100 format(' r1, r2, r3=          ',3(1x,f12.8))
 6050 format(' B1= ',e12.6,'   B2= ',e12.6,'   B3= ',e12.6)
 6000 format('Elondon       Ens         Eb1          Eb2     ',
     .       '  Eb4 ',/,    f8.6,4(1x,e12.6))
         write(6,*) ' --------------------------------------'
         write(6,*)
      end if
      IF(ideriv.eq.0)RETURN
C
C   DERIVATIVES
C E LONDON DERIVATIVES
C
      IF(XJ.EQ.0.0)GOTO 1
      XJS=(XJ1+XJ2+XJ3)/8.
      T1P=C*(1.+2.*A1*X(1))/EF1-F*T1
      T2P=C*(1.+2.*A1*X(2))/EF2-F*T2
      T3P=C*(1.+2.*A1*X(3))/EF3-F*T3
      ELON1P=(S1(2)+T1P)/2.-(S1(2)-T1P)*(.375*XJ1-XJS)/XJ
      ELON2P=(S2(2)+T2P)/2.-(S2(2)-T2P)*(.375*XJ2-XJS)/XJ
      ELON3P=(S3(2)+T3P)/2.-(S3(2)-T3P)*(.375*XJ3-XJS)/XJ
C
C    ENS DERIVATIVES
C
      ENSPWN=(AN1*WN*2.+AN2*3.*WN2+AN3*4.*WN3+AN4*5.*WN4)/EXNS
      ENSPR=(-3.)*FNS*R2*ENS
C
C   WN DERIVATIVES
C
      DELTA=-1.
      IF(WN.EQ.WNT)DELTA=1.
      WNP1=(2.*X(1)*(X(3)-X(2))+X22-X23)*DELTA
      WNP2=(2.*X(2)*(X(1)-X(3))+X23-X21)*DELTA
      WNP3=(2.*X(3)*(X(2)-X(1))+X21-X22)*DELTA
C
C   DENS/DXI=(DENS/DWN)(DWN/DXI)+(DENS/DR)(DR/DXI)
C
      ENSP1=ENSPWN*WNP1+ENSPR
      ENSP2=ENSPWN*WNP2+ENSPR
      ENSP3=ENSPWN*WNP3+ENSPR
C
C    WB DERIVATIVES
C
      WB1P=(X(1)/X(3)-1.)/X(2)-1./X(3)-(COS2+COS3)/X(1)
c     not used:
      W23P=(X(3)/X(2)-1.)/X(1)-1./X(2)-(COS1+COS2)/X(3)
      WB2P=(X(2)/X(3)-1.)/X(1)-1./X(3)-(COS1+COS3)/X(2)
C
      WB3P=(X(3)/X(2)-1.)/X(1)-1./X(2)-(COS1+COS2)/X(3)
C
C  DEB1/DX1=(DEB1/DWB)(DWB/DX1)+(DEB1/DR)(DR/DX1)
C
      EB1PR=WB*(F1*EB1T+F3*EB3T-B2/EXF1-2.*R*XL2/EXF3)
      EB1PWB=EB1T+EB3T
      EB1P1=EB1PWB*WB1P-EB1PR
      EB1P2=EB1PWB*WB2P-EB1PR
      EB1P3=EB1PWB*WB3P-EB1PR
      EB2PWB=(2.*WB*W1+3.*WB2*W2+4.*WB3*W3)/EXF2
      EB2PR=F2*(-2.)*R*EB2
      EB2P1=EB2PWB*WB1P+EB2PR
      EB2P2=EB2PWB*WB2P+EB2PR
      EB2P3=EB2PWB*WB3P+EB2PR
C
C     DEB4A/DXI=(DEB4A/DWB)(DWB/DXI)+(DEB4A/DRI)(DRI/DXI)
C                   +(DEB4A/DR)(DR/DXI)
C
      EB4APW=(D1/EXF1+2.*WB*D2/EXF2)*RI
      EB4APR=RI*(WB2*F2*(-2.)*R*D2/EXF2-F1*D1*WB/EXF1)
      EB4AP1=EB4APW*WB1P-EB4A/X21+EB4APR
      EB4AP2=EB4APW*WB2P-EB4A/X22+EB4APR
      EB4AP3=EB4APW*WB3P-EB4A/X23+EB4APR
      EB4BPW=EB4B*EQ
      B4BPEQ=EB4B*WB
      EB4BPR=EQ*WB*((-2.)*F2*R*D4/EXF2-F1*D3/EXF1)
      EB4BP1=EB4BPW*WB1P+B4BPEQ*(6.*X(1)-2.*R)+EB4BPR
      EB4BP2=EB4BPW*WB2P+B4BPEQ*(6.*X(2)-2.*R)+EB4BPR
      EB4BP3=EB4BPW*WB3P+B4BPEQ*(6.*X(3)-2.*R)+EB4BPR
      E1=ELON1P+ENSP1+EB1P1+EB2P1+EB4AP1+EB4BP1
      E2=ELON2P+ENSP2+EB1P2+EB2P2+EB4AP2+EB4BP2
      E3=ELON3P+ENSP3+EB1P3+EB2P3+EB4AP3+EB4BP3
      RETURN
    1 WRITE(6,2)
    2 FORMAT(1X,'  EQUILATERAL TRIANGLE,DERIVATIVES INFINITE ')
      E1=0.
      E2=0.
      E3=0.
      RETURN
      END
C
      SUBROUTINE VBIGR(X,S)
CCC****************************************************** module 21 ****
C    CALCULATES LONG RANGE POTENTIAL (FOR R > 10 A)
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/POTCOM/C6,C8,RKW(87),EKW(87),WKW(87)
      DIMENSION S(3)
      X2=X*X
      X3=X2*X
      X6=X3*X3
      C8A=C8/X2
      S(1)=-(C6+C8A)/X6
      S(2)=(C6*6.+C8A*8.)/X6/X
      S(3)=-(C6*42.+C8A*72.)/X6/X2
      RETURN
      END
C
      SUBROUTINE VH2(X,S1,S2,S3,isurf)
CCC**************************************************** module 22 **
C    THIS SUBROUTINE DECIDES WHICH OTHER SUBROUTINE TO CALL IN ORDER
C    TO EVALUATE THE ENERGY AND ITS DERIVATIVES AT DISTANCE=X
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/POTCOM/C6,C8,RKW(87),EKW(87),WKW(87)
      DIMENSION X(3),S1(3),S2(3),S3(3)
      if(isurf.eq.0)then
    1    IF(X(1).GT.10.)CALL VBIGR(X(1),S1)
         IF(X(1).GT.10.)GOTO 2
         CALL SPLID2(87,RKW,EKW,WKW,1,X(1),S1)
    2    IF(X(2).GT.10.)CALL VBIGR(X(2),S2)
         IF(X(2).GT.10.)GOTO 3
         CALL SPLID2(87,RKW,EKW,WKW,1,X(2),S2)
    3    IF(X(3).GT.10.)CALL VBIGR(X(3),S3)
         IF(X(3).GT.10.)RETURN
         CALL SPLID2(87,RKW,EKW,WKW,1,X(3),S3)
	 return
      end if
      if(isurf.eq.1)then
         write(6,*) ' vh2option removed '
c        id = 2
cx	 call vh2opt(x(1),s1,id)
cx	 call vh2opt(x(2),s2,id)
cx	 call vh2opt(x(3),s3,id)
	 return
      end if
      write(6,*) ' illegal value of isurf in subr.vh2, isurf=',isurf
      stop
      END
c
      BLOCK DATA LSTH2
CCC***************************************************** module 23 *****
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/POTCOM/C6,C8,RKW(87),EKW(87),WKW(87)
      COMMON/VCOM/C,A,A1,F,FN,FB1,FB2,FB3,XN1,XN2,XN3,XN4,B1,B2,B3
     *  ,G1,G2,G3,D1,D2,D3,D4,XL1,XL2
      DATA C6,C8/6.89992032,219.9997304/
      DATA C,A,A1,F/-1.2148730613,-1.514663474,-1.46,2.088442/
      DATA FN,XN1,XN2,XN3,XN4/.0035,.0012646477,-.0001585792,
     *   .0000079707,-.0000001151/
      DATA FB1,B1,B2/.52,3.0231771503,-1.08935219/
      DATA FB2,G1,G2,G3/.052,1.7732141742,-2.0979468223,-3.978850217/
      DATA D1,D2,D3,D4/.4908116374,-.8718696387,.1612118092,
     *  -.1273731045/
      DATA FB3,XL2,XL1/.79,.9877930913,-13.3599568553/
      DATA RKW/
     1 .4000000000D+00,.4500000000D+00,.5000000000D+00,.5500000000D+00,
     2 .6000000000D+00,.6500000000D+00,.7000000000D+00,.7500000000D+00,
     3 .8000000000D+00,.9000000000D+00,.1000000000D+01,.1100000000D+01,
     4 .1200000000D+01,.1300000000D+01,.1350000000D+01,.1390000000D+01,
     5 .1400000000D+01,.1401000010D+01,.1401099990D+01,.1410000000D+01,
     6 .1450000000D+01,.1500000000D+01,.1600000000D+01,.1700000000D+01,
     7 .1800000000D+01,.1900000000D+01,.2000000000D+01,.2100000000D+01,
     8 .2200000000D+01,.2300000000D+01,.2400000000D+01,.2500000000D+01,
     9 .2600000000D+01,.2700000000D+01,.2800000000D+01,.2900000000D+01,
     9 .3000000000D+01,.3100000000D+01,.3200000000D+01,.3300000000D+01,
     1 .3400000000D+01,.3500000000D+01,.3600000000D+01,.3700000000D+01,
     2 .3800000000D+01,.3900000000D+01,.4000000000D+01,.4100000000D+01,
     3 .4200000000D+01,.4300000000D+01,.4400000000D+01,.4500000000D+01,
     4 .4600000000D+01,.4700000000D+01,.4800000000D+01,.4900000000D+01,
     5 .5000000000D+01,.5100000000D+01,.5200000000D+01,.5300000000D+01,
     6 .5400000000D+01,.5500000000D+01,.5600000000D+01,.5700000000D+01,
     7 .5800000000D+01,.5900000000D+01,.6000000000D+01,.6100000000D+01,
     8 .6200000000D+01,.6300000000D+01,.6400000000D+01,.6500000000D+01,
     9 .6600000000D+01,.6700000000D+01,.6800000000D+01,.6900000000D+01,
     9 .7000000000D+01,.7200000000D+01,.7400000000D+01,.7600000000D+01,
     1 .7800000000D+01,.8000000000D+01,.8249999910D+01,.8500000000D+01,
     2 .9000000000D+01,.9500000000D+01,.1000000000D+02/
      DATA EKW/
     1  .879796188D+00, .649071056D+00, .473372447D+00, .337228924D+00,
     2  .230365628D+00, .145638432D+00, .779738117D-01, .236642733D-01,
     3 -.200555771D-01,-.836421044D-01,-.124538356D+00,-.150056027D+00,
     4 -.164934012D+00,-.172345701D+00,-.173962500D+00,-.174451499D+00,
     5 -.174474200D+00,-.174474400D+00,-.174474400D+00,-.174459699D+00,
     6 -.174055600D+00,-.172853502D+00,-.168579707D+00,-.162456813D+00,
     7 -.155066822D+00,-.146849432D+00,-.138131041D+00,-.129156051D+00,
     8 -.120123163D+00,-.111172372D+00,-.102412583D+00,-.939271927D-01,
     9 -.857809026D-01,-.780163108D-01,-.706699181D-01,-.637640270D-01,
     9 -.573117349D-01,-.513184414D-01,-.457831464D-01,-.407002530D-01,
     1 -.360577581D-01,-.318401624D-01,-.280271683D-01,-.245977718D-01,
     2 -.215296753D-01,-.187966785D-01,-.163688812D-01,-.142246837D-01,
     3 -.123370858D-01,-.106809878D-01,-.923028934D-02,-.796819096D-02,
     4 -.687029215D-02,-.591779314D-02,-.509229414D-02,-.437819496D-02,
     5 -.376259562D-02,-.323089623D-02,-.277399691D-02,-.237999732D-02,
     6 -.204229767D-02,-.175209799D-02,-.150299828D-02,-.128989853D-02,
     7 -.110689874D-02,-.949798920D-03,-.814999069D-03,-.700199190D-03,
     8 -.602999302D-03,-.516199400D-03,-.446599479D-03,-.386399548D-03,
     9 -.332799617D-03,-.290599668D-03,-.246599722D-03,-.215399753D-03,
     9 -.188899784D-03,-.143399836D-03,-.108599875D-03,-.867998994D-04,
     1 -.681999214D-04,-.527999393D-04,-.403999540D-04,-.313999636D-04,
     2 -.184999787D-04,-.120999861D-04,-.909998949D-05/
      DATA WKW/
     1  .308019605D+02, .214419954D+02, .154937452D+02, .115151545D+02,
     2  .871827707D+01, .673831756D+01, .527864661D+01, .419929947D+01,
     3  .333940643D+01, .219403463D+01, .149861953D+01, .103863661D+01,
     4  .730647471D+00, .518552387D+00, .441110777D+00, .383461006D+00,
     5  .373946396D+00, .358559402D+00, .372215569D+00, .356670198D+00,
     6  .312744133D+00, .261523038D+00, .180817537D+00, .124665543D+00,
     7  .807794104D-01, .486562494D-01, .251952492D-01, .452257820D-02,
     8 -.854560161D-02,-.196001146D-01,-.276538076D-01,-.344244662D-01,
     9 -.381080935D-01,-.421628973D-01,-.441600287D-01,-.454966841D-01,
     9 -.460129217D-01,-.458513118D-01,-.453815149D-01,-.440623159D-01,
     1 -.426089183D-01,-.404417185D-01,-.383839285D-01,-.361823035D-01,
     2 -.336666088D-01,-.302110314D-01,-.286090554D-01,-.255125522D-01,
     3 -.233005599D-01,-.201850499D-01,-.191990995D-01,-.161784216D-01,
     4 -.146071006D-01,-.126330766D-01,-.110605069D-01,-.996481997D-02,
     5 -.818014482D-02,-.765454189D-02,-.608163613D-02,-.575887028D-02,
     6 -.466284400D-02,-.408972107D-02,-.363824334D-02,-.295728079D-02,
     7 -.259261281D-02,-.221225014D-02,-.193837141D-02,-.203425060D-02,
     8 -.484614204D-03,-.226728547D-02,-.766232140D-03,-.307779418D-03,
     9 -.196264565D-02,+.131836977D-02,-.223083472D-02,-.750220030D-04,
     9 -.289074004D-03,-.220265690D-03,-.434861384D-03,+.971346041D-05,
     1 -.839919101D-04,-.153745275D-03,-.369227366D-04,-.249634065D-04,
     2 -.290482724D-04,-.148433244D-04,+.682166282D-05/
      END
