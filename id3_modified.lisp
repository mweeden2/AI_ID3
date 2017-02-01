;;; This file contains the modified or newly-written functions used by (run-id3) in
;;; "id3.lisp" for project 4 of iai, Fall 2015.

;===========================================================================================
;; This function was modified from the original.
(defun id3 (examples possible-attributes splitting-function)
  "A simple version of Quinlan's ID3 Program - see Machine Learning 1:1, 1986.                                                
   Numeric feature values, noisy data and missing values are not handled."

 ;;; This function produces a decision tree that classifies the examples                                                      
 ;;; provided, using these attributes.  The splitting function determines                                                     
 ;;; which attribute should be used to split a collection of + and - examples.                                                
 ;;; If all the examples are of the same type, a leaf node is returned.                                                       

 ;;; The resulting decision tree is a list of the form                                                                        
 ;;;     (attribute (value1 subtree1) (value2 subtree2) ... )                                                                 
 ;;;  or (decision  #-of-examples-in-training-set-located-here)                                                               
 ;;; In the first case, depending on the value of attribute, another decision                                                 
 ;;; tree must be traversed to make a decision.  In the second case, a                                                        
 ;;; decision is recorded, along with the number of examples from the                                                         
 ;;; training set that would be placed at this node.                                                                          

 ;;  See RUN-ID3 for a nice user interface to this function.                                                                  

 ;;  It is assumed that every example has a valid value for each attribute.                                                   
 ;;  The function VALIDATE-EXAMPLES can be used to pre-process examples.                                                      

     (cond ((null examples) '(? 0))  ; No more examples, an "undecided" node.                                                 
           ((all-positive? examples) `(+ , (length examples)))
           ((all-negative? examples) `(- , (length examples)))
           ((null possible-attributes) (error "Out of features - inconsistent data."))
           (t
            ;; choose attribute to branch on by calling choose-attribute                                                      
            (let* ((chosen-attribute (choose-attribute examples possible-attributes 'max-gain))
                   ;; remove the chosen attribute from the possible-attributes to get remaining attributes                    
                   (remaining-attributes (remove chosen-attribute possible-attributes))
                   (values (get-attribute-values chosen-attribute)))

              ;; return a list of the form                                                                                    
              ;;          (chosen-attribute (value1 subtree1) (value2 subtree2) ... )                                         
              ;; where the values are the different values that chosen-attribute can take                                     
              ;; and the subtrees are created by recursive calls to id3 where the first argument is                           
              ;; a subset of the examples that has the ith value for the attribute chosen-attribute                           
              ;; (this can be done by calling the function collect-examples-with-this-value),                                 
              ;; and the second argument is the remaining attributes as created above                                         
              (append (list chosen-attribute)
                      (construct-subtrees examples splitting-function chosen-attribute values remaining-attributes))))))

;===========================================================================================
;; This function was new-written
(defun construct-subtrees (examples splitting-function attribute attribute-values remaining-attributes
                                    &optional (subtrees nil))
  (let ((value (first attribute-values)))
    (if (eq 1 (length attribute-values)) ; if this is the last attribute value                                                
        (list (list value (id3 (collect-examples-with-this-value examples attribute value)
                         remaining-attributes splitting-function)))
      (append (list (list value (id3 (collect-examples-with-this-value examples attribute value)
                             remaining-attributes splitting-function)))
              (construct-subtrees examples splitting-function attribute
                                  (rest attribute-values) remaining-attributes)))))

(defun choose-attribute (examples attributes splitting-function)
  "Choose an attribute to split these examples into sub-groups.                                                               
The method of doing so is specified by the third argument."
  (case splitting-function
    (random      (nth (random (length attributes)) attributes)) ; make an arbitrary choice                                    
    (least-values (least_most attributes 'least-values))  ; choose the attribute with the least possible values               
    (most-values  (least_most attributes 'most-values))  ; choose the one with the most                                       
    (max-gain (max_gain examples attributes))  ; use Quinlan's gain measure (pg. 90) to choose                                
    (otherwise (error "ERROR - unknown splitting function"))))

;===========================================================================================
;; This function was newly-written
(defun max_gain (examples attributes &optional (gains nil))
  "Choose the attribute that represents the most information gain."
  (if (null gains)
      (max_gain examples attributes (get-gains examples attributes))
    (nth (index-of-max gains) attributes)))

;===========================================================================================
;; This function was newly-written
(defun get-gains (examples attributes &optional (gains nil) (count 0))
  "Calculate and collect the information gain measurement for each of the attributes."
  (if (eq (length gains) (length attributes))
      gains
    (let* ((this-attribute (nth count attributes))

           ;; get the probability of each value for this attribute based on example set                                       
           (probabilities (get-probabilities examples this-attribute))

           ;; entropy = -(P log2(P) + (1-P) log2(1-P))                                                                        
           (entropies (get-entropies probabilities))
           ;; weighted-entropy = (num-examples-with-this-value / num-examples) * entropy                                      
           (weighted-entropies (get-weighted-entropies examples this-attribute entropies))

           ;; class-proportion = p / (n + p)   - using the num of possitively/negatively-classified examples                  
           (num+ (count-positives examples))
           (num- (- (length examples) num+))
           (class-proportion (/ (coerce num+ 'float) (+ num- num+)))
           ;; goal-entropy = entropy(class-proportion)                                                                        
           (goal-entropy (- (+ (* class-proportion (log class-proportion 2))
                               (* (- 1 class-proportion) (log (- 1 class-proportion) 2))))))

      ;; gain = goal-entropy - sum(weighted-entropies)                                                                        
      (get-gains examples attributes
                 (append gains (list (- goal-entropy (apply #'+ weighted-entropies))))
                 (1+ count)))))

;===========================================================================================
;; This function was newly-writen
(defun get-probabilities (examples attribute &optional (values nil) (probabilities nil))
  "Find the probability of a positive classification given each of the attribute's values"
  (if (null values)
      (get-probabilities examples attribute (rest (assoc attribute *domains*)))
    (let* ((examples-with-value (collect-examples-with-this-value examples attribute (first values)))
           (num-pos-with-value (count-positives examples-with-value))
           (value-probability
            (if (null examples-with-value)
                0
              (/ num-pos-with-value (length examples-with-value)))))
      (if (eq 1 (length values))
          (append probabilities (list value-probability))
        (get-probabilities examples attribute (rest values)
                           (append probabilities (list value-probability)))))))

;===========================================================================================
;; This function was newly-written
(defun get-entropies (probabilities &optional (entropies nil))
  "Find the entropy associated with each attribute value probability"
  (if (null probabilities)
      entropies
    (let* ((p (first probabilities))
           ;; entropy = -(P log2(P) + (1-P) log2(1-P))                                                                        
           (entropy (if (or (< p 0.0000001) (> p 0.9999999))
                        0
                      (- (+ (* p (log p 2))
                            (* (- 1 p) (log (- 1 p) 2)))))))
      (get-entropies (rest probabilities) (append entropies (list entropy))))))

;===========================================================================================
;; This function was newly-written
(defun get-weighted-entropies (examples attribute entropies
                                        &optional (weighted-entropies nil) (count 0) &aux (values nil))
  "Find the weighted entropies for each attribute value based on how many examples have that attribute value."
  (if (null values)
      (setf values (rest (assoc attribute *domains*))))
  (if (null entropies)
      weighted-entropies
    (let ((value-entropy (first entropies))
          (num-ex-with-value
           (length (collect-examples-with-this-value examples attribute (nth count values))))
          (num-examples (length examples)))
      ;; weighted-entropy = (num-examples-with-this-value / num-examples) * entropy                                           
      (get-weighted-entropies examples attribute (rest entropies)
                              (append weighted-entropies
                                      (list (* value-entropy (/ num-ex-with-value num-examples))))
                              (1+ count)))))

;===========================================================================================
;; This function was nearly newly-written (the comments and skeleton were not newly-written)
(defun make-decision (example &optional (decision-tree *current-tree*) )
  "Use this decision tree to classify this unclassified instance."
  ;; Find the root attribute of the decision-tree                                                                             
  (let* ((root-attribute (first decision-tree))
         ;; For that attribute, find the attribute value in the example                                                       
         (root-value (second (assoc root-attribute  example)))
         ;; Find the subtree in decision-tree corresponding to that value                                                     
         (subtree (first (rest (assoc root-value (rest decision-tree))))))
    ;; If this subtree is a leaf node (will look like (+) or (-)) return that classification                                  
    (cond ((eq '+ (first subtree))
           '+)
          ((eq '- (first subtree))
           '-)
          ;; If the leaf node looks like (?), use the following code                                                          
          ((eq '? (first subtree))
           (if (> (count-matching-leaves decision-tree  '-)
                  (count-matching-leaves decision-tree '+))
               '-
             '+))
          ;; Otherwise, call make-decision recursively with the subtree identified above                                      
          (t
            (make-decision example subtree)))))

;===========================================================================================
;; This function was newly-created
(defun index-of-max (lst &optional (max 0) (count 0) (index 0))
  (if (null lst)
      index
    (if (> (first lst) max)
        (index-of-max (rest lst) (first lst) (1+ count) count)
      (index-of-max (rest lst) max (1+ count) index))))

;===========================================================================================
;; This function was slightly modified (the default value for the example-file arg was changed
;; from "CATEGORIZED.DATA" to "in.data").
(defun run-id3 (&optional (examples *train-examples*) (splitting-function 'random)
                          (examples-file "in.data")
                          (report-tree? *trace-id3*)
                &aux start-time)
  "Check these examples for correctness, build a decision tree, then                                                          
draw the tree (if requested) and, finally, report some statistics about it."
  (when (null examples)
    (format t "~%~%Constructing the test set ... ")
    (build-example-lists examples-file)
    (format t " ~D training examples produced." (length *train-examples*))
    (setf examples *train-examples*))
  (format t "~%~%Building Decision Tree ...")
  (if (validate-examples examples)
     (progn (setf start-time (get-internal-run-time))
            (setf *current-tree* (id3 examples *all-attributes* splitting-function))
            (format t " finished in ~,3F sec.~%"
                    (convert-to-sec (- (get-internal-run-time) start-time)))
            (if report-tree? (print-decision-tree))
            (let ( (interior-nodes (count-interior-nodes *current-tree*))
                   (leaf-nodes     (count-leaf-nodes *current-tree*)))
              (format t "~%~%Tree size=~A    interior nodes=~A  leaf-nodes=~A~%"
                 (+ interior-nodes leaf-nodes) interior-nodes leaf-nodes))
            (format t "   positive leaves=~A  negative leaves=~A  undecided leaves=~A ~%~%"
              (count-matching-leaves *current-tree* '+)
              (count-matching-leaves *current-tree* '-)
              (count-matching-leaves *current-tree* '?))
            (measure-correctness-ID3 *test-examples*))
     (format t "~%~% RUN ABORTED DUE TO ERRONEOUS TRAINING DATA.~%~%")))