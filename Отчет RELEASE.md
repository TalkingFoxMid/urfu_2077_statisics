![[Pasted image 20220601030926.png|700x1000]]
<div style="page-break-after: always;"></div>

## 1. Корреляционная таблица:
![[Pasted image 20220601031754.png]]

## 2. Одномерные распределения статистических признаков:
![[Pasted image 20220531020838.png]]

![[Pasted image 20220531020847.png]]

## 3. Числовые характеристики X и Y:
### Выборочное среднее:
Выборочное среднее:
$\overline{x_в} = \sum_{i=1}^{l}{x_i \cdot {n_{x_i} \over n}}$
$\overline{x_в} = -272.65$
$\overline{y_в} = 548.419$

### Выборочная дисперсия:
$D_{xв} = \sum_{i=1}^{l}{(x_i-\overline{x_в}){n_{x_i} \over n}}$
$D_{xв} =89.65$
$D_{yв} =340.11$


### Выборочное СКО:
$\sigma_{xв} = \sqrt{D_{xв}}$
$D_{xв} =9.468$
$D_{yв} =18.44$

### "Исправленная" Дисперсия:
$\S^2_{x} = {n \over n-1}D_{xв}$
$\S^2_{x} = 89.65 * 93/92 = 90.61$
$\S^2_{y} = 340.11 * 93/92 = 343.77$

### "Исправленное" СКО :
$s_{x} = \sqrt{S^2{x}}$
$s_{x} = \sqrt(90.61) = 9.52$
$s_{y} = \sqrt(343.77) = 18.54$

## 4. Построить гистограммы и полигоны относительных частот X и Y
### Гистограммы:

![[Python/results/Hist X.png]]
![[Python/results/Hist Y.png]]

### Полигоны:

![[Python/results/Polygones X.png]]
![[Python/results/Polygones Y.png]]

### Гистограммы и Полигоны:

![[Python/results/Hist_Polygones X.png]]
![[Python/results/Hist_Polygones Y.png]]

## 5.  Найти точечные оценки генеральных параметров a и σ. Найти интервальные оценки генеральных параметров a и σ с доверительной вероятностью.

Для X:
Точечные оценки:
$\overline{x_в} = -272.65$
$s_{x} = \sqrt(90.61) = 9.52$

Интервальные оценки:

Найдем $t_{\gamma}$, коэффицент Стьюдента. А также $q(\gamma, n)$.

$\gamma = 1 - \alpha = 0.95$


$t_{\gamma} = \Phi^{-1}(\gamma / 2)=\Phi^{-1}(0.95 / 2)=1.96$.

По таблице:
$q = 0.151$

Найдем доверительный интервал для $a$:
$$[\overline{x} - {t_{\gamma}s_x \over \sqrt{n}}, \overline{x} + {t_{\gamma}s_x \over \sqrt{n}}]$$

$${t_{\gamma}s_x \over \sqrt{n}} = {1.96 \cdot 9.52 \over \sqrt{93}} = 0.20$$

Интервал: $$[-272.85, -272.45]$$

Найдем доверительный интервал для $\sigma$:
$$[s_x(1-q), s_x(1+q)]$$

$$(9.52\cdot(1-0.151))=8.082$$
$$(9.52\cdot(1+0.151))=10.957$$

Интервал: $$[8.082, 10.957]$$

## 6. Используя критерий согласия Пирсона, проверить гипотезу о нормальном распределении генеральной совокупности признака X при уровне значимости a=0.05 . Записать вид теоретической плотности вероятности и теоретической функции распределения. Построить их графики. Построить в одной плоскости полигоны эмпирических и теоретических частот признака X.

<div style="page-break-after: always;"></div>

1) Частоты первого и последнего интервала малы, а потому объедим их с соседними. Новая таблица корреляции:

| Y\X            | $$[-290, -282)$$ | $$[-282, -274)$$ | $$[-274, -266)$$ | $$[-266, -258)$$ | $$[-258, -250)$$ | $$n_{y_i}$$ |
| -------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------- |
| $$[494, 509)$$ |                  |                  |                  |                  |        2          | 2           |
| $$[509, 524)$$ |                  |                  |                  | 2                | 3                | 5           |
| $$[524, 539)$$ |                  |                  | 7                | 12               |                  | 19          |
| $$[539, 554)$$ |                  | 7                | 21               | 4                |                  | 32          |
| $$[554, 569)$$ | 3                | 17               | 3                |                  |                  | 23          |
| $$[569, 584)$$ | 9                |                  |                  |                  |                  | 9           |
| $$[584, 599)$$ | 3                |                  |                  |                  |                  | 3           |
| $$n_{x_i}$$    | 15               | 24               | 31               | 18               | 5                |             |


Вспомним полученные значения:
> $s_{x}=9.52$
> $\overline{x}=-272.65$
> $n = 93$

2) Расчетная таблица:

| №   | $\alpha_{i}$ - $\alpha_{i-1}$ | $n_{x_i}$ | $u_{i}={\alpha_{i-1}-\overline{x} \over s_x}$ | $\Phi(u_i)$ | $p'_i= \Delta \Phi$ | $n'_{x_i} = n * p'_i$ |   Целое $n'_{x_i}$  | $(n_{x_i}-n'_{x_i}) \over n'_{x_i}$ |
| --- | ----------------------------- | --------- | --------------------------------------------- | ----------- | ------------------- | ---------- | --- | ----------------------------------- |
|0|[-298, -282)|15|-1,822|-0,466|0,129|12,006|12|0,750
|1|[-282, -274)|24|-0,982|-0,337|0,281|26,114|26|0,154
|2|[-274, -266)|31|-0,142|-0,056|0,314|29,174|29|0,138
|3|[-266, -258)|18|0,699|0,258|0,180|16,759|17|0,059
|4|[-258, -242)|5|1,539|0,438|0,053|4,938|5|0,000
|$\sum$| |93| | | | | 89|1.1006007177406771|


Наблюдаемое значение статистики $\chi^2_{набл}$ = 1.10

Число степеней свободы: $k = l' - 3 = 5 - 3 = 2$

По таблице: $\chi^2_{кр}= 6$

Так как $\chi^2_{набл} < \chi^2{кр}$, нет оснований отвергать гипотезу

Теоретическая плотность вероятности:
$$f(x)={1 \over 9.52 \cdot \sqrt{2 \pi}} e^{-{({x+272.65})^2 \over 181.2}}$$
![[Pasted image 20220531212512.png]]
Теоретическая функция распределения:
$$F(x) = {1 \over 2}({1 + erf({x+272.65 \over \sqrt{181.2}})})$$
![[Pasted image 20220531212751.png]]
Полигоны эмпирических и теоретических частот:

![[Python/results/Hist_Norm_X.png]]
## 7.  Вычислить коэффициент корреляции* X и Y. Проверить значимость коэффициента корреляции, используя критерий согласия Стьюдента.

Выборочный корреляционный момент $K_{xy} = \overline{XY} - \overline{X_v}\cdot\overline{Y_v}$, где
$$\overline{XY} = {\sum\sum{X_i Y_i n_{ij}}\over n}$$

> $X_i$ - середина отрезка i по x
> $Y_i$ - середина полуинтервала i по y

$K_{XY}= -159.08$
$r_{XY} = {K_{XY} \over s_x \cdot s_y}=-0.90$

#### Проверка по критерию Стьюдента:
$$t_{ex} = {r\sqrt{n - 2} \over \sqrt{1-r^2}} = -19.84$$

$$t_{cr} = 1.99$$ - по таблице

$t$ ожидаемое больше чем $t$ критическое, из чего следует: что коэффицент корреляции значим.


## 8-9. Найти уравнения теоретических линий регрессии X на Y и Y на X. Построить их графики.
$r_{XY}=-0.90$
$s_x = 9.52$ 
$s_y = 18.542$
$\overline{X_v}=-272.65$
$\overline{Y_v}=548.419$

#### Y на X:
Теоретическое уравнение:

$$y = {r_{XY} \cdot s_y \over s_x} \cdot x + \overline{Y_v} - {r_{XY} \cdot s_y \over s_x}\cdot \overline{X_v}$$


$y=-1.76x+71.19$

Эмпирические линии:

| ${X_i}$                | -294   | -286   | -278   | -270   | -262   | -254   | -246   |
| ---------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Эмпирическое значение  | 591.5  | 572.2  | 557    | 544.56 | 533.1  | 516.5  |   501.5     |
| Теоретическое значение | 588.63 | 574.55 | 560.47 | 546.39 | 532.31 | 518.23 | 504.15 |

![[Python/results/Regression X to Y.png]]

#### X на Y:
Теоретическое уравнение:

$$x = {r_{XY} \cdot s_x \over s_y} \cdot y + \overline{X_v} - {r_{XY} \cdot s_x \over s_y}\cdot \overline{Y_v}$$

$x=-0.46y-18.15$

Эмпирические линии:

| ${Y_i}$                | 501.5   | 516.5   | 531.5   | 546.5   | 561.5   | 576.5   | 591.5   |
| ---------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Эмпирическое значение  | -246    | -257.2  | -264.94 | -270.75 | -278    | -286    | -288.6        |
| Теоретическое значение | -248.84 | -255.74 | -262.64 | -269.54 | -276.44 | -283.34 | -290.24 |

![[Python/results/Regression Y to X.png]]

<div style="page-break-after: always;"></div>
## 10.  Сформировать таблицу результатов (на отдельной странице сразу после титульного листа, то есть это будет _в т о р а я_ страница отчета; на третью страницу не залезать. Две таблицы на одной странице. И номер варианта):


Величина| Значение
:----------------|-------------:
Выборочная средняя признака X| $-272.65$
Выборочная дисперсия признака X| $89.65$
Выборочное СКО признака X| $9.46$
Исправленное СКО признака X| $9.52$
Мода признака X| $$[-274, -266)$$ 
Медиана признака X| $$[-274, -266)$$
Выборочная средняя признака Y| $548.419$
Выборочная дисперсия признака Y| $340.11$
Выборочное СКО признака Y| $18.442$
Исправленное СКО признака Y| $18.542$
Мода признака Y| $$[539, 554)$$
Медиана признака Y| $$[539, 554)$$
Наблюдаемое значение критерия Пирсона | $$1.10$$
Доверительный интервал для a | $[-272.85, -272.45]$
Доверительный интервал для  σ | $[8.082, 10.957]$
Ковариация | $116.440$
Коэффициент корреляции | $0.65$
Наблюдаемое значение критерия Стьюдента | $8.37$
Уравнение теоретической линии регрессии Y на X | $y=-1.76x+71.19$
Уравнение теоретической линии регрессии X  на Y | $x=-0.46y-18.15$