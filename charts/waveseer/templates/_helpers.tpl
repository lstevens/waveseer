{{/*
Return the name of the chart
*/}}
{{- define "waveseer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Return the fullname of the chart
*/}}
{{- define "waveseer.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "waveseer.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
