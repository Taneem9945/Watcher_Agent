# Addressing Remaining Challenges

These notes summarize how the current prototype will address the remaining mixed-data Watcher Agent challenges.

## Missing Data

- Missing data should be explicit, not silently hidden.
- Events can keep nullable fields such as `src_ip`, `dst_ip`, `username`, `uri`, or `signature`.
- Each event can include `missing_fields` and `available_fields` so the LLM knows what evidence exists.
- Mamba cannot understand missingness automatically because it only receives numbers.
- For Mamba, missingness should be encoded with flags such as `src_ip_present`, `username_present`, or `uri_present`.
- This lets missingness become useful signal instead of being confused with a real zero value.

## Source Meaning

- Source meaning will be handled first by preserving `source_type` on every event.
- Example source types include `zeek_conn`, `zeek_http`, `zeek_dns`, `zeek_ssh`, `suricata_alert`, `host_auth`, `host_syslog`, `endpoint_telemetry`, and `pcap_packet`.
- Adding `source_type` does not create more rows. It adds one field to each existing row.
- Combining multiple data sources creates more rows, but `source_type` keeps the combined stream understandable.
- For the LLM, `source_type` can be shown directly as text.
- For Mamba, `source_type` should be encoded numerically, such as one-hot or embedding-based encoding.
- A deeper taxonomy such as `evidence_level` or `sensor_type` can be added later if needed, but it is not required for the first prototype.

## Feature Encoding

- The shared event shape may contain strings, IPs, ports, URLs, usernames, hashes, commands, and free-text messages.
- Mamba needs consistent numeric vectors, so these fields must be encoded before reaching Mamba.
- Numeric fields can be normalized.
- Categorical fields such as `source_type`, protocol, service, and event type can be one-hot encoded or embedded.
- Text fields such as URI, command, or log message can start with simple features such as length, keyword flags, or suspicious-token counts.
- More advanced text embeddings can be added later if the prototype needs richer semantic input.

### P025 Encoding Issues

- Zeek connection and HTTP logs include IP addresses such as `id.orig_h` and `id.resp_h`. These should not be converted into one large raw number. Better prototype features include `src_ip_internal`, `dst_ip_internal`, `same_subnet`, `src_host_id`, `dst_host_id`, and common port flags.

- Zeek HTTP logs include URI strings such as `/tomcatwar.jsp?pwd=j&cmd=cat /etc/passwd`. These are important, but Mamba cannot read the raw string directly. Better prototype features include `uri_length`, `has_query_string`, `has_cmd_param`, `has_passwd_keyword`, `has_shadow_keyword`, `path_depth`, and file-extension flags such as `file_extension_jsp`.

- HTTP logs include user agents such as `Wget/1.21.2` and `python-requests/2.32.3`. Better prototype features include `user_agent_wget`, `user_agent_python_requests`, `user_agent_browser_like`, `user_agent_cli_tool`, and `user_agent_missing`.

- Zeek file logs include hashes such as `md5`, `sha1`, and `sha256`. The raw hash string is not useful as a direct numeric feature. Better prototype features include `file_hash_present`, `seen_bytes`, `total_bytes`, MIME type flags, `file_seen_before`, and `same_hash_repeated`.

- Host logs such as `auth.log` include free-text messages about user creation, password changes, sudo sessions, and commands. Better prototype features include `event_user_created`, `event_password_changed`, `event_sudo_used`, `event_ssh_keygen`, `username_present`, and `command_present`.

- Behavior telemetry such as `bt.jsonl` includes messages like service startup, HTTP API operations, and actions such as `HideInterface`. Better prototype features include `event_service_started`, `event_hide_interface`, `event_http_operation`, and source/component indicators.

- Suricata `eve.json` can include deeply nested JSON statistics and alerts. These should be flattened into fields such as `suricata_kernel_packets`, `suricata_kernel_drops`, `suricata_alert_count`, `suricata_http_flow_count`, and `suricata_ssh_flow_count`.

### Prototype Encoding Fix

- Keep raw readable strings for the LLM evidence packet.

- Convert raw fields into simple numeric features for Mamba.

- Add missingness flags so missing data is visible to the model.

- Start with security-relevant flags and counts instead of full text embeddings.

- Use source and event encodings such as `source_type_zeek_http`, `source_type_host_auth`, and `event_type_http_request`.

- Use relationship and behavior encodings such as `src_ip_internal`, `dst_port_common_http`, `uri_has_cmd`, `uri_has_passwd`, `event_user_created`, and `suricata_alert_count`.

- Use presence encodings such as `uri_present`, `username_present`, `hash_present`, and `src_ip_present`.

- Treat advanced text embeddings as optional future work, not a blocker for the first prototype.

### Stable Facts And Mamba Signal

- Stable identifiers such as IP addresses, hostnames, usernames, connection UIDs, file IDs, and hashes should remain preserved outside the Mamba representation.

- Mamba should not be responsible for remembering what an IP address literally is. It should learn how that IP's behavior changes across the sequence.

- The input encoding for an event stays fixed before Mamba. For example, the same source IP should map to the same encoded identity or stable feature each time it appears.

- Mamba's hidden representation can change the contextual meaning of that event based on previous events, but it should not replace the original event facts.

- This avoids losing stationary datapoints while still allowing Mamba to model evolving behavior.

- The architecture should keep two tracks:

  - Stable event facts: timestamps, IPs, hosts, usernames, UIDs, hashes, and source types.

  - Mamba learned signal: sequence representation, embedding summary, top activations, and embedding change over time.

- The LLM packet should include both tracks when needed: readable stable facts for grounding and Mamba sequence signal for temporal interpretation.

- This is a strong direction for the mixed-data Watcher Agent because it lets IPs and identities stay recognizable while Mamba captures behavior over time.

## Window Design

- Events should be grouped into time windows after normalization.
- For the current prototype, windowing is simplified because the mixed dataset already provides timestamps.
- The current default assumption is to use fixed 60-second windows.
- The window choice affects what the system can detect.
- Short windows are better for bursts and fast attacks.
- Longer windows are better for slower multi-step behavior.
- Overlapping windows can help preserve context between adjacent time periods.
- Window size still needs to be revisited later, but it is not a blocker for the first prototype.

## Event Relationship Building

- Related events should be linked using shared fields such as host, source IP, destination IP, username, connection UID, timestamp, or file hash.
- This helps the LLM see an incident chain instead of isolated facts.
- Example relationship: Zeek HTTP request, Suricata alert, and host auth event involving the same host in the same time range.
- Relationship summaries can be added to the context packet for LLM reasoning.

## Stateful Operation

- Stateful operation is considered handled for the prototype through app-level stream memory.
- Current stream memory tracks recent windows, last signal, last delta, rolling trends, embedding L2 norm history, previous Mamba representation summaries, window count, and last window ID.
- This is not true recurrent Mamba hidden-state carryover, but it is sufficient for the current prototype.

## Already Scoped Assumptions

- Duplicate evidence is not treated as a major prototype issue.
- Duplicate rows are expected to be handled during shared-shape normalization where possible.
- Timestamp alignment is assumed to exist for testing in the current mixed dataset.
- Different granularity is not considered a blocker because each source can become timestamped evidence.
- Volume is not treated as a blocker for the first prototype.
- Mamba labeling is not part of the current direction.
- Formal evaluation is out of scope for the immediate prototype.
- Trust and explainability are not primary constraints right now.
- Current context selection is assumed to be good enough for the prototype.
