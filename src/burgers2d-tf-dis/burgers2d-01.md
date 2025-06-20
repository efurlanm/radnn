existe uma diferença conceitual significativa entre burgers2d.py e discovery1d.py além da dimensionalidade.

  `discovery1d.py` implementa uma abordagem de Physics-Informed Neural Networks (PINN) para a descoberta de parâmetros.
  Nele:
   * A solução da PDE (u(x,t)) é aproximada por uma rede neural.
   * As derivadas da PDE (e, consequentemente, o resíduo da equação) são calculadas usando diferenciação automática
     através da rede neural.
   * A função de perda inclui termos que minimizam tanto o erro em relação aos dados medidos quanto o resíduo da equação 
     de Burgers, garantindo que a solução da rede neural satisfaça a física do problema.
   * Os parâmetros da PDE (lambda_1 e lambda_2) são tratados como variáveis treináveis da rede neural e são descobertos
     durante o processo de otimização.

  `burgers2d-01.py`, por outro lado, utiliza uma abordagem mais tradicional de otimização de parâmetros com um resolvedor 
  numérico explícito (diferenças finitas). Nele:
   * A simulação direta (forward_simulation) é realizada por um método de diferenças finitas explícito, que já incorpora
     a física da equação de Burgers 2D.
   * Os "dados medidos" são gerados a partir desta simulação direta com um nu_real conhecido.
   * O parâmetro nu é uma variável TensorFlow que é otimizada para minimizar a diferença (erro quadrático médio) entre a
     saída da simulação de diferenças finitas (com o nu_guess atual) e os dados "medidos".
   * A diferenciação automática é usada para calcular os gradientes da perda em relação a nu_guess, mas a simulação em si
     não é uma rede neural.

  Em resumo:

   * discovery1d.py é um exemplo de PINN, onde a rede neural aprende a solução e os parâmetros da PDE simultaneamente,
     incorporando a equação diretamente na função de perda.
   * burgers2d.py é um exemplo de otimização de parâmetros baseada em modelo, onde um resolvedor numérico tradicional é
     usado para a simulação direta, e a diferenciação automática é empregada para otimizar um parâmetro do modelo contra
     dados.

  Para que burgers2d.py seja uma "ampliação" de discovery1d.py no sentido conceitual de PINN, ele precisaria ser
  reestruturado para usar uma rede neural para aproximar as soluções u(x,y,t) e v(x,y,t) e incorporar os resíduos das
  equações de Burgers 2D na função de perda, permitindo a descoberta de nu (e potencialmente outros parâmetros) como
  variáveis treináveis da rede neural.
 
