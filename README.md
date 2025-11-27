 $ 
 
python main.py   

--- uruchamia synthetic/render_scene.py, który tworzy:
obraz rgb_0001.png,
prawdziwą mapę głębi depth_gt_0001.npy

--- midas_wrapper/run_midas.py przewiduje głębię z RGB
– zapisuje wynik jako depth_midas_0001.npy

---Porównuje MiDaS z prawdziwą głębią
– evaluation/evaluate_midas.py dopasowuje skalę MiDaS → ground truth
– liczy metryki: RMSE, AbsRel, δ<1.25
– zapisuje wyniki oraz zeskalowaną mapę głębi

 python evaluation/select_point.py - do mierzenia odleglosci na obrazie na podstawie midasa

 
