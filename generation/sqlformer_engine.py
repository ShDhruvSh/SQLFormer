class SQLFormerEngine:
    def __init__(self, model, fsm, logits_processor, schema):
        self.model = model
        self.fsm = fsm
        self.logits = logits_processor
        self.schema = schema

    def generate(self, prompt):
        sql_tokens = {"select": [], "from": [], "where": [], "group_by": [], "order_by": []}
        state = self.fsm.state
        step = 0

        while not self.fsm.completed and step < 25:
            step += 1
            allowed_next = self.fsm.allowed_next()

            if state == "START":
                candidates = ["SELECT"]
            elif state == "SELECT":
                possible_columns = [c for table in self.schema["tables"] for c in self.schema["columns"][table]]
                print("[DEBUG] Extracted possible columns:", possible_columns)
                if sql_tokens["select"]:
                    candidates = possible_columns + ["FROM"]
                else:
                    candidates = possible_columns
                print("[DEBUG] Candidates for SELECT:", candidates)
            elif state == "FROM":
                candidates = self.schema["tables"] + ["WHERE", "GROUP_BY", "ORDER_BY", "END"]
            elif state == "WHERE":
                candidates = [c for table in self.schema["tables"] for c in self.schema["columns"][table]] + ["GROUP_BY", "ORDER_BY", "END"]
            elif state == "GROUP_BY":
                candidates = [c for table in self.schema["tables"] for c in self.schema["columns"][table]] + ["ORDER_BY", "END"]
            elif state == "ORDER_BY":
                candidates = [c for table in self.schema["tables"] for c in self.schema["columns"][table]] + ["END"]
            else:
                candidates = allowed_next

            filtered = self.logits.mask(candidates, state, sql_tokens)
            if state == "SELECT":
                print("[DEBUG] Filtered tokens for SELECT:", filtered)
            if not filtered:
                break  # No valid tokens left

            if self.model:
                prompt_text = self._format_prompt(prompt, sql_tokens, state)
                logits = self.model.next_token_logits(prompt_text)
                best_idx = self._pick_best(logits, filtered)
                token = filtered[best_idx]
            else:
                token = filtered[0]

            # Add token to SQL construction, if not duplicated
            if state == "SELECT" and token != "FROM":
                if token not in sql_tokens["select"]:
                    sql_tokens["select"].append(token)
            elif state == "FROM" and token not in ["WHERE", "GROUP_BY", "ORDER_BY", "END"]:
                if token not in sql_tokens["from"]:
                    sql_tokens["from"].append(token)
            elif state == "WHERE" and token not in ["GROUP_BY", "ORDER_BY", "END"]:
                if token not in sql_tokens["where"]:
                    sql_tokens["where"].append(token)
            elif state == "GROUP_BY" and token not in ["ORDER_BY", "END"]:
                if token not in sql_tokens["group_by"]:
                    sql_tokens["group_by"].append(token)
            elif state == "ORDER_BY" and token != "END":
                if token not in sql_tokens["order_by"]:
                    sql_tokens["order_by"].append(token)

            state = self.fsm.advance(token)

        query = self._build_sql(sql_tokens)
        return query

    def _format_prompt(self, nl_prompt, sql_tokens, state):
        sql_str = self._build_sql(sql_tokens)
        return f"{nl_prompt}\nCurrent SQL: {sql_str}\nNext clause: {state}"

    def _build_sql(self, sql_toks):
        out = "SELECT " + ", ".join(sql_toks["select"])
        if sql_toks["from"]:
            out += " FROM " + ", ".join(sql_toks["from"])
        if sql_toks["where"]:
            out += " WHERE " + " AND ".join([f"{c} = ?" for c in sql_toks["where"]])
        if sql_toks["group_by"]:
            out += " GROUP BY " + ", ".join(sql_toks["group_by"])
        if sql_toks["order_by"]:
            out += " ORDER BY " + ", ".join(sql_toks["order_by"])
        return out

    def _pick_best(self, logits, allowed_tokens):
        """
        Given model logits (Tensor) and allowed_tokens (list of str), find the best allowed token.
        - Map allowed_tokens to token_ids using the model's tokenizer
        - Mask logits for only allowed token ids
        - Return the index of the best allowed token in allowed_tokens
        """
        import torch
        # Convert allowed_tokens to ids using the model's tokenizer
        allowed_ids = [self.model.tokenizer.convert_tokens_to_ids(token) 
                       for token in allowed_tokens]
        logits = logits.squeeze(0)  # (vocab_size,)
        mask = torch.full_like(logits, fill_value=-float('inf'))
        for idx, token_id in enumerate(allowed_ids):
            if token_id is not None and token_id >=0 and token_id < logits.size(0):
                mask[token_id] = logits[token_id]
        best_token_id = torch.argmax(mask).item()
        for i, token in enumerate(allowed_tokens):
            if self.model.tokenizer.convert_tokens_to_ids(token) == best_token_id:
                return i
        return 0